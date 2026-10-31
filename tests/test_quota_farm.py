"""Offline regression tests: all provider calls are intercepted by MockTransport."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from quota_farm.client import ChatClient, ProviderError, image_payload
from quota_farm.cli import verify_run
from quota_farm.config import Budget, Profile, load_spec
from quota_farm.engine import Runner, judge, read_dataset
from quota_farm.export import export_jsonl, materialize, safe_relative_path
from quota_farm.ledger import BudgetExceeded, Ledger


def profile(**overrides):
    data = {
        "model": "fake-model", "base_url": "https://api.openai.com/v1",
        "api_key_env": "TEST_QUOTA_FARM_KEY", "billing": "paid",
        "reserve_input_tokens": 4000, "max_output_tokens": 800,
        "usd_per_m_input": 1.0, "usd_per_m_output": 4.0,
    }
    data.update(overrides)
    return Profile.parse("mock", data)


def make_spec(root: Path, *, mode="generate_review", reviews=2):
    (root / "dataset.jsonl").write_text(json.dumps(
        {"id": "sample-1", "input": "Produce safe training datum", "context": "synthetic fixture",
         "candidate": "example" if mode == "evaluate" else ""}
    ) + "\n", encoding="utf-8")
    (root / "generate.md").write_text("Generate an artifact for ${input}", encoding="utf-8")
    (root / "review.md").write_text("Review ${candidate} for ${input}", encoding="utf-8")
    reviewers = [{"id": f"critic-{i}", "model": "critic", "prompt": "review.md",
                  "min_score": 0.8} for i in range(reviews)]
    manifest = {
        "version": 1, "name": "offline-test", "mode": mode, "dataset": "dataset.jsonl",
        "output_dir": "runs", "concurrency": 2, "max_records": 3,
        "models": {"gen": {
            "model": "fake", "base_url": "https://api.openai.com/v1",
            "api_key_env": "TEST_QUOTA_FARM_KEY", "billing": "paid",
            "reserve_input_tokens": 4096, "max_output_tokens": 800,
            "usd_per_m_input": 1.0, "usd_per_m_output": 4.0,
        }, "critic": {
            "model": "fake", "base_url": "https://api.openai.com/v1",
            "api_key_env": "TEST_QUOTA_FARM_KEY", "billing": "paid",
            "reserve_input_tokens": 4096, "max_output_tokens": 800,
            "usd_per_m_input": 1.0, "usd_per_m_output": 4.0,
        }},
        "reviewers": reviewers,
        "budgets": {"daily_premium_tokens": 250000, "daily_secondary_tokens": 2500000,
                    "monthly_paid_usd": 1, "run_paid_usd": 0.5},
    }
    if mode == "generate_review":
        manifest["generator"] = {"model": "gen", "prompt": "generate.md"}
    (root / "job.json").write_text(json.dumps(manifest), encoding="utf-8")
    return load_spec(root / "job.json")


class ConfigurationTests(unittest.TestCase):
    def test_incentive_host_and_pool_guard(self):
        with self.assertRaises(ValueError):
            profile(billing="free_expected", pool="secondary",
                    base_url="https://example.com/v1")
        with self.assertRaises(ValueError):
            profile(billing="free_expected")
        free = profile(billing="free_expected", pool="secondary")
        self.assertEqual(free.pool, "secondary")

    def test_requires_explicit_paid_price(self):
        with self.assertRaises(ValueError):
            profile(usd_per_m_input=0)

    def test_http_restricted_to_loopback(self):
        with self.assertRaises(ValueError):
            profile(base_url="http://example.com/v1")
        self.assertEqual(profile(base_url="http://localhost:8000/v1", billing="local",
                                 api_key_env=None).billing, "local")

    def test_reject_unsafe_paths(self):
        for name in ("../bad", "/etc/passwd", "C:/windows/system.ini", "a/../../b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative_path(name)
        self.assertEqual(str(safe_relative_path("levels/first.txt")), "levels/first.txt")

    def test_review_fail_closed(self):
        self.assertEqual(judge('{"score":0.9,"verdict":"accept","findings":[]}')["score"], 0.9)
        for invalid in ('{"score":true,"verdict":"accept","findings":[]}',
                        '{"score":1.2,"verdict":"accept","findings":[]}',
                        '{"score":0.9,"verdict":"maybe","findings":[]}',
                        'garbage'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                judge(invalid)

    def test_dataset_path_confined(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = make_spec(root)
            row = {"id": "sample-1", "input": "hello", "images": ["../outside.png"]}
            (root / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_dataset(spec)


class LedgerTests(unittest.TestCase):
    def test_reservation_settlement_and_ceiling(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder))
            spec = Budget(premium_tokens=5000, secondary_tokens=10000,
                          monthly_usd=0.01, run_usd=0.01, cycle_start_day=1)
            free = profile(billing="free_expected", pool="premium",
                           reserve_input_tokens=2000, max_output_tokens=500)
            one = ledger.reserve("run", free, spec)
            two = ledger.reserve("run", free, spec)
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("run", free, spec)
            ledger.finish(one, free, {"prompt_tokens": 100, "completion_tokens": 20},
                          success=True)
            ledger.finish(two, free, None, success=False)
            summary = ledger.summary("run")
            self.assertEqual(summary["premium"]["uncertain_requests"], 1)
            self.assertEqual(summary["premium"]["reserved_or_used_tokens"], 2620)

    def test_paid_run_limit_and_no_refund_on_ambiguous_error(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder))
            budget = Budget(250000, 2500000, 10.0, 0.0065, 1)
            paid = profile(reserve_input_tokens=2000, max_output_tokens=1000)
            t = ledger.reserve("run", paid, budget)
            ledger.finish(t, paid, None, success=False)
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("run", paid, budget)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_tool_fields_and_usage(self):
        captured = []
        def handler(request):
            captured.append(json.loads(request.content))
            self.assertEqual(request.headers["Authorization"], "Bearer fake-token")
            return httpx.Response(200, json={
                "id": "receipt-1",
                "choices": [{"message": {"role": "assistant", "content": "result"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            })
        with patch.dict(os.environ, {"TEST_QUOTA_FARM_KEY": "fake-token"}):
            async with ChatClient(httpx.MockTransport(handler)) as client:
                result = await client.complete(profile(), "System", "User")
        self.assertEqual(result.content, "result")
        self.assertEqual(result.usage["completion_tokens"], 5)
        self.assertTrue(captured[0]["store"] is False)
        self.assertFalse(any(key in captured[0] for key in ("tools", "functions", "tool_choice")))

    async def test_vision_data_and_rejection_of_tool_calls(self):
        captured = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"mockpixels")
            def handler(request):
                payload = json.loads(request.content)
                captured.append(payload)
                return httpx.Response(200, json={
                    "choices": [{"message": {"role": "assistant", "content": None,
                                            "tool_calls": [{"id": "forbidden"}]},
                                 "finish_reason": "tool_calls"}]
                })
            with patch.dict(os.environ, {"TEST_QUOTA_FARM_KEY": "fake-token"}):
                async with ChatClient(httpx.MockTransport(handler)) as client:
                    with self.assertRaises(ProviderError):
                        await client.complete(profile(image_token_reserve=1200), "sys", "inspect", [path])
            content = captured[0]["messages"][1]["content"]
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertFalse("tools" in captured[0])

    async def test_missing_key_fails_without_http(self):
        async with ChatClient(httpx.MockTransport(lambda _: self.fail("network used"))) as client:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ProviderError):
                    await client.complete(profile(), "sys", "user")


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_generation_multi_review_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = make_spec(root)
            seen = []
            artifact = json.dumps({"files": [{"path": "script/intro.txt", "encoding": "utf8",
                                            "content": "hello-level"}]})
            def handler(request):
                body = json.loads(request.content)
                seen.append(body)
                prompt = body["messages"][1]["content"]
                result = ('{"score":0.93,"verdict":"accept","findings":[]}'
                          if prompt.startswith("Review") else artifact)
                return httpx.Response(200, json={
                    "id": "fake-1", "choices": [{"message": {"content": result},
                                              "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 95, "completion_tokens": 35},
                })
            with patch.dict(os.environ, {"TEST_QUOTA_FARM_KEY": "fake-token"}):
                async with ChatClient(httpx.MockTransport(handler)) as client:
                    runner = Runner(spec, Ledger(root / "ledger"), client)
                    run_dir = await runner.run(read_dataset(spec))
            self.assertEqual(len(seen), 3)
            report = verify_run(run_dir)
            self.assertTrue(report["evidence_verified"])
            self.assertEqual(report["summary"]["counts"], {"accepted": 1})
            count = export_jsonl(run_dir, root / "corpus.jsonl")
            self.assertEqual(count, 1)
            self.assertIn("script/intro.txt", (root / "corpus.jsonl").read_text())
            files = materialize(run_dir, root / "artifacts")
            self.assertEqual(files, 1)
            self.assertEqual((root / "artifacts" / "script" / "intro.txt").read_text(), "hello-level")
            with self.assertRaises(FileExistsError):
                materialize(run_dir, root / "artifacts")
            item = next((run_dir / "items").glob("*.json"))
            modified = json.loads(item.read_text())
            modified["candidate"] = "tampered"
            item.write_text(json.dumps(modified))
            # Candidate editing isn't cryptographically signed; source/response edits are detected.
            modified["source"]["input"] = "tampered"
            item.write_text(json.dumps(modified))
            self.assertFalse(verify_run(run_dir)["evidence_verified"])

    async def test_independent_reviewer_disagreement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = make_spec(root, mode="evaluate")
            calls = 0
            def handler(_):
                nonlocal calls
                calls += 1
                verdict = "accept" if calls == 1 else "reject"
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps({
                        "score": 0.95 if calls == 1 else 0.2,
                        "verdict": verdict, "findings": []})}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                })
            with patch.dict(os.environ, {"TEST_QUOTA_FARM_KEY": "fake-token"}):
                async with ChatClient(httpx.MockTransport(handler)) as client:
                    path = await Runner(spec, Ledger(root / "state"), client).run(read_dataset(spec))
            result = json.loads((path / "items" / "sample-1.json").read_text())
            self.assertEqual(result["status"], "rejected")
            self.assertTrue(result["disagreement"])

    async def test_free_incentive_requires_explicit_acknowledgment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = make_spec(root)
            raw = json.loads((root / "job.json").read_text())
            raw["models"]["gen"]["billing"] = "free_expected"
            raw["models"]["gen"]["pool"] = "secondary"
            (root / "job.json").write_text(json.dumps(raw))
            spec = load_spec(root / "job.json")
            async with ChatClient(httpx.MockTransport(lambda _: self.fail("network"))) as client:
                runner = Runner(spec, Ledger(root / "ledger"), client)
                with patch.dict(os.environ, {"TEST_QUOTA_FARM_KEY": "fake-token"}):
                    with self.assertRaisesRegex(ValueError, "confirm-incentive"):
                        runner.preflight(read_dataset(spec))


if __name__ == "__main__":
    unittest.main()
