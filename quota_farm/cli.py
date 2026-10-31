"""Small JSON CLI for coding supervisors and scheduled workflows."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from .client import ChatClient
from .config import load_spec
from .engine import Runner, canonical, read_dataset, sha, verify_item
from .export import export_jsonl, materialize
from .ledger import Ledger


def _state_dir() -> Path:
    state = os.environ.get("QUOTA_FARM_STATE_DIR")
    if state:
        return Path(state).expanduser()
    home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return home / "quota-farm"


def verify_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    errors = []
    if sha(canonical(manifest)) != summary.get("source_manifest_sha256"):
        errors.append("manifest digest differs from run summary")
    counts: Counter = Counter()
    items = list((run_dir / "items").glob("*.json"))
    expected_items = summary.get("item_sha256", {})
    if set(expected_items) != {path.stem for path in items}:
        errors.append("item index/expected digests mismatch")
    for path in items:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            counts[item["status"]] += 1
            if path.stem != item["id"]:
                errors.append(f"{path.name}: identifier mismatch")
            if sha(canonical(item)) != expected_items.get(path.stem):
                errors.append(f"{path.name}: original item digest mismatch")
            verify_item(item)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.name}: {error}")
    if len(items) != summary.get("total") or dict(counts) != summary.get("counts"):
        errors.append("item count/status mismatch")
    return {"summary": summary, "evidence_verified": not errors, "evidence_errors": errors}


async def _run(args: argparse.Namespace) -> Path:
    spec = load_spec(args.spec)
    ledger = Ledger(Path(args.state_dir).expanduser())
    rows = read_dataset(spec)
    async with ChatClient() as client:
        runner = Runner(spec, ledger, client, confirm_incentive=args.confirm_incentive)
        return await runner.run(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quota-farm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="validate inputs, estimate call count; no network or secret required")
    plan.add_argument("spec")

    run = commands.add_parser("run", help="execute with SQLite quota reservations and immutable item receipts")
    run.add_argument("spec")
    run.add_argument("--state-dir", default=str(_state_dir()))
    run.add_argument("--confirm-incentive", action="store_true",
                     help="I manually verified complimentary billing on this OpenAI API project")

    report = commands.add_parser("report", help="inspect completed run and verify local evidence hashes")
    report.add_argument("run_dir")

    export = commands.add_parser("export-jsonl", help="export accepted text and reviewer decisions")
    export.add_argument("run_dir")
    export.add_argument("--output", required=True)

    material = commands.add_parser("materialize", help="write accepted artifact manifests to a local destination")
    material.add_argument("run_dir")
    material.add_argument("--output", required=True)
    material.add_argument("--confirm-write", action="store_true",
                          help="write generated candidate files; never overwrite existing files")

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            spec = load_spec(args.spec)
            rows = read_dataset(spec)
            used = {r.model for r in spec.reviewers}
            if spec.generator_model:
                used.add(spec.generator_model)
            info = {
                "job": spec.name, "mode": spec.mode, "records": len(rows),
                "expected_inference_calls": len(rows) *
                    (len(spec.reviewers) + int(bool(spec.generator_model))),
                "models": {name: {
                    "model": spec.models[name].model, "billing": spec.models[name].billing,
                    "pool": spec.models[name].pool
                } for name in sorted(used)},
                "warning": "This is not a billing or API eligibility verification. "
                           "Free token requests may be charged; image tokens need measured calibration.",
            }
            print(json.dumps(info, indent=2))
        elif args.command == "run":
            folder = asyncio.run(_run(args))
            print(json.dumps({
                "run_dir": str(folder),
                **verify_run(folder),
            }, indent=2))
        elif args.command == "report":
            result = verify_run(Path(args.run_dir))
            print(json.dumps(result, indent=2))
            return 0 if result["evidence_verified"] else 1
        elif args.command == "export-jsonl":
            count = export_jsonl(Path(args.run_dir), Path(args.output))
            print(json.dumps({"accepted_records_exported": count, "output": args.output}))
        elif args.command == "materialize":
            if not args.confirm_write:
                raise ValueError("--confirm-write is required")
            count = materialize(Path(args.run_dir), Path(args.output))
            print(json.dumps({"files_written": count, "output": args.output}))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"quota-farm: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
