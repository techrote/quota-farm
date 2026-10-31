"""Strict, declarative job definitions. Secrets are environment-only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

IDENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _positive_int(value: object, name: str, *, minimum: int = 1, maximum: int = 10000000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}: expected integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, name: str, *, minimum: float = 0, maximum: float = 1000000) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: invalid numeric value")
    return float(value)


def _file(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("paths must be nonempty relative filenames")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"file not present inside spec directory: {name}")
    return path


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not IDENT.fullmatch(value):
        raise ValueError(f"{description}: expected portable identifier")
    return value


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    base_url: str
    key_env: str | None
    billing: str
    pool: str | None
    reserve_input_tokens: int
    max_output_tokens: int
    image_token_reserve: int
    usd_per_m_input: float
    usd_per_m_output: float
    timeout_s: float
    max_attempts: int
    max_images: int
    max_image_bytes: int
    token_parameter: str

    @classmethod
    def parse(cls, name: str, data: dict) -> "Profile":
        if not isinstance(data, dict):
            raise ValueError(f"models.{name}: expected object")
        model = data.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"models.{name}.model: required")
        base = str(data.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        parsed = urlparse(base)
        loopback = parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if parsed.query or parsed.fragment or parsed.username or parsed.password or (
            parsed.scheme != "https" and not (parsed.scheme == "http" and loopback)
        ) or not parsed.hostname or not base.endswith("/v1"):
            raise ValueError(f"models.{name}: HTTPS /v1 endpoint required (HTTP only for loopback)")
        billing = data.get("billing")
        if billing not in ("free_expected", "paid", "local"):
            raise ValueError(f"models.{name}.billing: free_expected, paid or local required")
        pool = data.get("pool")
        if billing == "free_expected":
            if base != "https://api.openai.com/v1" or pool not in ("premium", "secondary"):
                raise ValueError("free_expected must use api.openai.com and an explicit pool")
        elif pool is not None:
            raise ValueError("only free_expected models can specify a pool")
        key_env = data.get("api_key_env")
        if key_env is not None and (not isinstance(key_env, str) or not ENV_NAME.fullmatch(key_env)):
            raise ValueError(f"models.{name}.api_key_env: expected environment variable name")
        if billing != "local" and not key_env:
            raise ValueError(f"models.{name}: paid/free calls require api_key_env")
        input_rate = _number(data.get("usd_per_m_input", 0), "usd_per_m_input")
        output_rate = _number(data.get("usd_per_m_output", 0), "usd_per_m_output")
        if billing == "paid" and (input_rate <= 0 or output_rate <= 0):
            raise ValueError("paid model profiles require explicit positive rates")
        token_param = data.get("token_parameter", "max_completion_tokens")
        if token_param not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("token_parameter must be max_completion_tokens or max_tokens")
        return cls(
            name=_identifier(name, "model profile name"), model=model.strip(), base_url=base,
            key_env=key_env, billing=billing, pool=pool,
            reserve_input_tokens=_positive_int(data.get("reserve_input_tokens"), "reserve_input_tokens"),
            max_output_tokens=_positive_int(data.get("max_output_tokens"), "max_output_tokens", maximum=100000),
            image_token_reserve=_positive_int(data.get("image_token_reserve", 0), "image_token_reserve", minimum=0),
            usd_per_m_input=input_rate, usd_per_m_output=output_rate,
            timeout_s=_number(data.get("timeout_s", 90), "timeout_s", minimum=1, maximum=600),
            max_attempts=_positive_int(data.get("max_attempts", 1), "max_attempts", maximum=4),
            max_images=_positive_int(data.get("max_images", 8), "max_images", maximum=24),
            max_image_bytes=_positive_int(data.get("max_image_bytes", 4000000), "max_image_bytes", maximum=8000000),
            token_parameter=token_param,
        )


@dataclass(frozen=True)
class Reviewer:
    identifier: str
    model: str
    prompt: Path
    min_score: float
    include_images: bool


@dataclass(frozen=True)
class Budget:
    premium_tokens: int
    secondary_tokens: int
    monthly_usd: float
    run_usd: float
    cycle_start_day: int

    @classmethod
    def parse(cls, data: dict) -> "Budget":
        if not isinstance(data, dict):
            raise ValueError("budgets must be an object")
        monthly = _number(data.get("monthly_paid_usd", 0), "monthly_paid_usd")
        run = _number(data.get("run_paid_usd", 0), "run_paid_usd")
        if run > monthly:
            raise ValueError("run_paid_usd cannot exceed monthly_paid_usd")
        return cls(
            _positive_int(data.get("daily_premium_tokens", 250000), "daily_premium_tokens"),
            _positive_int(data.get("daily_secondary_tokens", 2500000), "daily_secondary_tokens"),
            monthly, run,
            _positive_int(data.get("cycle_start_day", 1), "cycle_start_day", maximum=28),
        )


@dataclass(frozen=True)
class Spec:
    path: Path
    name: str
    mode: str
    dataset: Path
    output_dir: Path
    concurrency: int
    max_records: int
    models: dict[str, Profile]
    generator_model: str | None
    generator_prompt: Path | None
    reviewers: tuple[Reviewer, ...]
    budget: Budget


def load_spec(filename: str | Path) -> Spec:
    path = Path(filename).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("version 1 JSON object required")
    root = path.parent
    mode = raw.get("mode")
    if mode not in ("generate_review", "evaluate"):
        raise ValueError("mode must be generate_review or evaluate")
    profiles = raw.get("models")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("models must be a nonempty object")
    models = {name: Profile.parse(name, settings) for name, settings in profiles.items()}
    gen = raw.get("generator")
    gen_model = None
    gen_prompt = None
    if mode == "generate_review":
        if not isinstance(gen, dict) or gen.get("model") not in models:
            raise ValueError("generate_review requires generator.model")
        gen_model = gen["model"]
        gen_prompt = _file(root, gen.get("prompt"))
    elif gen is not None:
        raise ValueError("evaluate mode cannot specify a generator")
    reviewers = raw.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers or len(reviewers) > 8:
        raise ValueError("1 to 8 independent reviewers required")
    reviewers_parsed = []
    seen = set()
    for item in reviewers:
        if not isinstance(item, dict) or item.get("model") not in models:
            raise ValueError("each reviewer must reference a configured model")
        ident = _identifier(item.get("id"), "reviewer id")
        if ident in seen:
            raise ValueError("reviewer IDs must be unique")
        seen.add(ident)
        if type(item.get("include_images", False)) is not bool:
            raise ValueError("include_images must be boolean")
        reviewers_parsed.append(Reviewer(
            ident, item["model"], _file(root, item.get("prompt")),
            _number(item.get("min_score", 0.75), "min_score", maximum=1),
            item.get("include_images", False),
        ))
    output = raw.get("output_dir", "runs")
    if not isinstance(output, str) or not output or Path(output).is_absolute():
        raise ValueError("output_dir must be relative to the spec")
    output_path = (root / output).resolve()
    if not output_path.is_relative_to(root):
        raise ValueError("output_dir escapes spec directory")
    return Spec(
        path, _identifier(raw.get("name"), "name"), mode, _file(root, raw.get("dataset")),
        output_path,
        _positive_int(raw.get("concurrency", 4), "concurrency", maximum=32),
        _positive_int(raw.get("max_records", 100), "max_records", maximum=10000),
        models, gen_model, gen_prompt, tuple(reviewers_parsed),
        Budget.parse(raw.get("budgets", {})),
    )
