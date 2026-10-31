"""A deliberately tool-free OpenAI-compatible Chat Completions transport."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Profile


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    content: str
    usage: dict | None
    receipt: dict


def image_payload(path: Path, profile: Profile) -> tuple[dict, dict]:
    data = path.read_bytes()
    if not data or len(data) > profile.max_image_bytes:
        raise ValueError(f"image exceeds allowed size or empty: {path}")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(f"unsupported image magic bytes: {path}")
    uri = "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
    return (
        {"type": "image_url", "image_url": {"url": uri}},
        {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
    )


def text_byte_upper_bound(system: str, user: str) -> int:
    # Conservative for standard text tokenisers; not a guarantee across providers.
    return len(system.encode("utf-8")) + len(user.encode("utf-8")) + 512


class ChatClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(transport=transport, trust_env=False)

    async def __aenter__(self) -> "ChatClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def complete(
        self, profile: Profile, system: str, user: str, images: list[Path] | None = None
    ) -> Completion:
        images = images or []
        if len(images) > profile.max_images:
            raise ValueError(f"image limit exceeded for {profile.name}")
        if images and profile.image_token_reserve <= 0:
            raise ValueError("image_token_reserve must be configured for vision requests")
        if text_byte_upper_bound(system, user) > profile.reserve_input_tokens:
            raise ValueError(
                f"{profile.name}: text exceeds reserve_input_tokens; raise the reservation"
            )
        content = [{"type": "text", "text": user}]
        hashes = []
        for path in images:
            item, info = image_payload(path, profile)
            content.append(item)
            hashes.append(info)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content if images else user}]
        # No tools, functions, web search, file search or tool choice are submitted.
        payload = {
            "model": profile.model, "messages": messages, "stream": False,
            profile.token_parameter: profile.max_output_tokens,
        }
        if profile.base_url == "https://api.openai.com/v1":
            payload["store"] = False
        headers = {"Content-Type": "application/json"}
        if profile.key_env:
            key = os.environ.get(profile.key_env)
            if not key:
                raise ProviderError(f"missing API key environment variable: {profile.key_env}")
            headers["Authorization"] = "Bearer " + key
        started = time.monotonic()
        for attempt in range(profile.max_attempts):
            try:
                response = await self._client.post(
                    profile.base_url + "/chat/completions",
                    headers=headers, json=payload, timeout=profile.timeout_s,
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < profile.max_attempts:
                    await asyncio.sleep(min(2 ** attempt, 5))
                    continue
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                if choice.get("finish_reason") == "tool_calls" or choice["message"].get("tool_calls"):
                    raise ProviderError("unexpected tool call in tool-free response")
                answer = choice["message"].get("content")
                if isinstance(answer, list):
                    answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
                if not isinstance(answer, str) or not answer.strip():
                    raise ProviderError("provider returned empty/refused completion")
                usage = data.get("usage")
                receipt = {
                    "model": profile.model, "profile": profile.name, "billing": profile.billing,
                    "pool": profile.pool, "provider_id": data.get("id"),
                    "finish_reason": choice.get("finish_reason"),
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "images": hashes,
                    "usage": usage if isinstance(usage, dict) else None,
                    "attempts": attempt + 1,
                }
                return Completion(answer, usage if isinstance(usage, dict) else None, receipt)
            except (httpx.TimeoutException, httpx.TransportError) as error:
                if attempt + 1 == profile.max_attempts:
                    raise ProviderError(f"transport failure (charge uncertain): {type(error).__name__}") from error
                await asyncio.sleep(min(2 ** attempt, 5))
            except httpx.HTTPStatusError as error:
                # Avoid exposing provider response bodies that can echo private inputs.
                raise ProviderError(f"provider HTTP {error.response.status_code}") from error
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ProviderError("provider returned unsupported response structure") from error
        raise ProviderError("request exhausted attempts")
