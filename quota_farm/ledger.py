"""Cross-process quota reservations with an auditable SQLite journal.

Reservations are deliberately conservative: ambiguous provider failures are NOT
refunded, because a timed-out inference may have incurred billable tokens.
These are local limits, not authoritative OpenAI billing measurements.
"""
from __future__ import annotations

import calendar
from contextlib import contextmanager
from typing import Iterator
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Budget, Profile


class BudgetExceeded(RuntimeError):
    pass


def _cycle(now: datetime, day: int) -> str:
    year, month = now.year, now.month
    if now.day < day:
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _micros(inputs: int, outputs: int, profile: Profile) -> int:
    # Rate is dollars per million tokens. Multiply tokens by rate to obtain microdollars.
    return math.ceil(inputs * profile.usd_per_m_input + outputs * profile.usd_per_m_output)


@dataclass(frozen=True)
class Ticket:
    identifier: str
    bucket: str
    reserved_input: int
    reserved_output: int


class Ledger:
    def __init__(self, state_dir: Path):
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = state_dir / "ledger.sqlite3"
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, bucket TEXT NOT NULL,
                period TEXT NOT NULL, amount INTEGER NOT NULL, spent INTEGER,
                reserved_input INTEGER NOT NULL, reserved_output INTEGER NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS requests_bucket ON requests(bucket, period)")
            db.execute("CREATE INDEX IF NOT EXISTS requests_run ON requests(run_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS limits (
                bucket TEXT NOT NULL, period TEXT NOT NULL, ceiling INTEGER NOT NULL,
                PRIMARY KEY(bucket, period)
            )""")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        # The sqlite3.Connection context manager does NOT close the handle.
        # Explicit closing is mandatory on Windows before a temporary state directory
        # can be removed, and prevents long-lived file locks on every platform.
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            db.execute("PRAGMA busy_timeout=15000")
            db.execute("PRAGMA journal_mode=WAL")
            yield db
        finally:
            db.close()

    def reserve(
        self, run_id: str, profile: Profile, budget: Budget, *, image_count: int = 0
    ) -> Ticket:
        inp = profile.reserve_input_tokens + profile.image_token_reserve * image_count
        out = profile.max_output_tokens
        bucket = ("paid" if profile.billing == "paid" else
                  profile.pool if profile.billing == "free_expected" else "local")
        ticket = Ticket(uuid.uuid4().hex, bucket, inp, out)
        if bucket == "local":
            return ticket
        now = datetime.now(timezone.utc)
        period = _cycle(now, budget.cycle_start_day) if bucket == "paid" else now.date().isoformat()
        amount = _micros(inp, out, profile) if bucket == "paid" else inp + out
        limit = (math.floor(budget.monthly_usd * 1000000) if bucket == "paid"
                 else budget.premium_tokens if bucket == "premium" else budget.secondary_tokens)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                # A single state directory must not silently mix different monthly
                # grant renewal days; distinct projects may use distinct state dirs.
                if bucket == "paid":
                    previous_cycle = db.execute(
                        "SELECT value FROM metadata WHERE key='cycle_start_day'"
                    ).fetchone()
                    if previous_cycle and previous_cycle[0] != str(budget.cycle_start_day):
                        raise ValueError("inconsistent cycle_start_day for shared ledger")
                    db.execute(
                        "INSERT OR IGNORE INTO metadata VALUES('cycle_start_day', ?)",
                        (str(budget.cycle_start_day),)
                    )
                # Never let another job with a larger config raise the day's cap.
                prior = db.execute(
                    "SELECT ceiling FROM limits WHERE bucket=? AND period=?",
                    (bucket, period)
                ).fetchone()
                limit = min(limit, prior[0]) if prior else limit
                db.execute(
                    "INSERT INTO limits(bucket, period, ceiling) VALUES(?,?,?) "
                    "ON CONFLICT(bucket,period) DO UPDATE SET ceiling=excluded.ceiling",
                    (bucket, period, limit)
                )
                current = db.execute(
                    "SELECT COALESCE(SUM(COALESCE(spent, amount)), 0) "
                    "FROM requests WHERE bucket=? AND period=?",
                    (bucket, period),
                ).fetchone()[0]
                if current + amount > limit:
                    raise BudgetExceeded(
                        f"{bucket} allowance exhausted/insufficient: {current}+{amount}>{limit} ({period})"
                    )
                if bucket == "paid":
                    run_amount = db.execute(
                        "SELECT COALESCE(SUM(COALESCE(spent, amount)), 0) "
                        "FROM requests WHERE bucket='paid' AND run_id=?", (run_id,)
                    ).fetchone()[0]
                    if run_amount + amount > math.floor(budget.run_usd * 1000000):
                        raise BudgetExceeded("per-run paid USD ceiling reached")
                db.execute(
                    "INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (ticket.identifier, run_id, bucket, period, amount, None, inp, out,
                     "reserved", now.isoformat()),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return ticket

    def finish(
        self, ticket: Ticket, profile: Profile, usage: dict | None, *, success: bool
    ) -> None:
        if ticket.bucket == "local":
            return
        # Missing usage or uncertain failures retain their full reservation.
        spent = None
        if success and isinstance(usage, dict):
            in_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
            out_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
            if type(in_tokens) is int and type(out_tokens) is int and in_tokens >= 0 and out_tokens >= 0:
                spent = (_micros(in_tokens, out_tokens, profile)
                         if ticket.bucket == "paid" else in_tokens + out_tokens)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE requests SET spent=?, status=? WHERE id=? AND status='reserved'",
                (spent, "settled" if success else "uncertain", ticket.identifier),
            )
            db.execute("COMMIT")

    def summary(self, run_id: str) -> dict:
        with self._db() as db:
            rows = db.execute(
                "SELECT bucket, COUNT(*), SUM(COALESCE(spent,amount)), "
                "SUM(CASE WHEN status='uncertain' THEN 1 ELSE 0 END) "
                "FROM requests WHERE run_id=? GROUP BY bucket", (run_id,)
            ).fetchall()
        result = {}
        for bucket, count, charged, uncertain in rows:
            result[bucket] = {
                "requests": count,
                "reserved_or_used_tokens": charged if bucket != "paid" else None,
                "accounted_usd": round(charged / 1000000, 6) if bucket == "paid" else None,
                "uncertain_requests": uncertain,
            }
        return result
