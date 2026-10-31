"""Deterministic extraction of reviewed content. Never execute model output."""
from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_FILES = 1000
WINDOWS_BAD = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
               *(f"LPT{i}" for i in range(1, 10))}


def accepted(run_dir: Path) -> list[dict]:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file() or not (run_dir / "summary.json").is_file():
        raise ValueError("not a completed quota-farm run")
    items = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted((run_dir / "items").glob("*.json"))]
    return [item for item in items if item.get("status") == "accepted"
            and item.get("accepted") is True]


def export_jsonl(run_dir: Path, destination: Path) -> int:
    items = accepted(run_dir)
    with destination.open("x", encoding="utf-8") as output:
        for item in items:
            output.write(json.dumps({
                "id": item["id"], "candidate": item["candidate"],
                "source_sha256": item["source_sha256"],
                "reviews": {
                    name: info["decision"] for name, info in item["reviews"].items()
                },
            }, ensure_ascii=False, sort_keys=True) + "\n")
    return len(items)


def safe_relative_path(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/"):
        raise ValueError("artifact path must be safe relative POSIX path")
    path = PurePosixPath(raw)
    if any(part in ("", ".", "..") or ":" in part or part.endswith(" ")
           or part.endswith(".") or part.split(".")[0].upper() in WINDOWS_BAD
           for part in path.parts):
        raise ValueError(f"unsafe artifact path: {raw}")
    return path


def materialize(run_dir: Path, destination: Path) -> int:
    """Explicit local write; no overwrite and no permission to change repo sources."""
    items = accepted(run_dir)
    dest = destination.resolve()
    if dest.exists() and (dest.is_symlink() or not dest.is_dir()):
        raise ValueError("destination must be directory")
    planned: list[tuple[PurePosixPath, bytes]] = []
    names = set()
    for item in items:
        try:
            payload = json.loads(item["candidate"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{item['id']}: candidate must be a JSON artifact manifest") from error
        files = payload.get("files")
        if not isinstance(files, list) or not files or len(files) > 16:
            raise ValueError(f"{item['id']}: expected 1-16 files")
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("files entries must be objects")
            path = safe_relative_path(entry.get("path"))
            if path in names:
                raise ValueError(f"duplicate output path: {path}")
            names.add(path)
            if entry.get("encoding") == "utf8" and isinstance(entry.get("content"), str):
                content = entry["content"].encode("utf-8")
            elif entry.get("encoding") == "base64" and isinstance(entry.get("content"), str):
                try:
                    content = base64.b64decode(entry["content"], validate=True)
                except binascii.Error as error:
                    raise ValueError(f"invalid base64: {path}") from error
            else:
                raise ValueError("encoding must be utf8 or base64 with string content")
            if len(content) > MAX_FILE_BYTES:
                raise ValueError(f"artifact too large: {path}")
            planned.append((path, content))
    if len(planned) > MAX_FILES or sum(len(b) for _, b in planned) > MAX_TOTAL_BYTES:
        raise ValueError("materialization exceeds total size/file count limits")
    # Prevalidate every path before the first write; reject preexisting/symlink targets.
    for path, _ in planned:
        full = dest.joinpath(*path.parts)
        if full.exists() or full.is_symlink():
            raise FileExistsError(full)
        parent = full.parent
        while parent != dest:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f"unsafe parent directory: {parent}")
            parent = parent.parent
    dest.mkdir(parents=True, exist_ok=True)
    for path, content in planned:
        full = dest.joinpath(*path.parts)
        full.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(full, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
    return len(planned)
