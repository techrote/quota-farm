# quota-farm

A small, **tool-free inference worker farm** for a more capable coding supervisor (OpenCode, OMP, Codex or a custom orchestrator). Batches are ordinary text/image Chat Completions; Python handles the files, queueing, budgets and evidence. No worker can call a filesystem, shell, web search, MCP or hosted tool.

**Status:** v0.1 implementation candidate, source-only/offline tests. **Do not interpret the configured free pools as guaranteed free billing.**

## Quick start

Python 3.11+:

```bash
python -m pip install -e .
export OPENAI_API_KEY=...  # Windows PowerShell: $env:OPENAI_API_KEY="..."
quota-farm plan examples/sprocket/job.json
# FIRST verify the API project's free data-sharing incentive on a tiny test call,
# then review your OpenAI Usage Dashboard before using this explicit acknowledgment:
quota-farm run examples/sprocket/job.json --confirm-incentive
quota-farm report examples/sprocket/runs/sprocket-corpus-RUN-ID
quota-farm export-jsonl examples/sprocket/runs/sprocket-corpus-RUN-ID --output /tmp/accepted.jsonl
```

The first `run` outputs an actual `run_dir` that can be passed to `report` or `export-jsonl`. For local/on-prem or paid-only providers, `--confirm-incentive` is unnecessary. Production API credentials are never needed in the offline test suite or `plan`.

## Model and provider configuration

Every job is a versioned JSON manifest. Each named profile specifies:

- `base_url` ending with `/v1`, `model`, `api_key_env` (name of an environment variable, **not the key**), and `billing`: `free_expected`, `paid`, or `local`.
- For `free_expected`, the explicit `premium`/`secondary` token pool and the canonical OpenAI API endpoint. This is accounting intent **only**, not evidence that the actual requests qualify.
- A conservative `reserve_input_tokens` upper reservation, `max_output_tokens` bound including reasoning, and optionally `image_token_reserve` per image. Vision estimates must be calibrated against provider usage.
- For `paid`, positive `usd_per_m_input` and `usd_per_m_output` rates in dollars per million tokens. Set model- and service-tier-specific prices yourself, including potential surcharges; no automatic vendor price lookup.
- An optional `token_parameter: "max_tokens"` for NVIDIA or local endpoints that do not accept `max_completion_tokens` (default). No temperature or unsupported reasoning parameters are forced onto generic endpoints.

Example alternative model profile (replace a job's model profile):

```json
{
  "model": "your-nim-model",
  "base_url": "https://integrate.api.nvidia.com/v1",
  "api_key_env": "NVIDIA_API_KEY",
  "billing": "paid",
  "usd_per_m_input": 0.5,
  "usd_per_m_output": 1.5,
  "reserve_input_tokens": 8000,
  "max_output_tokens": 1000,
  "token_parameter": "max_tokens"
}
```

**The prices above are placeholders**, not NVIDIA pricing. For a local OpenAI-compatible server, use `http://127.0.0.1:8000/v1` and `billing:"local"`, omit `api_key_env`, and configure your actual model name. HTTPS is required outside loopback.

## Three pipelines and direct asset authoring

| Pipeline | Example | What the worker sees | What is produced |
|---|---|---|---|
| Sprocket synthetic corpus | `examples/sprocket/job.json` | Synthetic specifications, candidate text | Generator draft + independently scored format and semantic reviews |
| Artwork / level review | `examples/artwork/job.json` | Text description and *optional local PNG/JPG/WebP screenshots* | Independent visual and playability critiques |
| CyberSand NPC assessment | `examples/npc/job.json` | Timestamped frame captures and separate text telemetry | Visual behaviour critique + telemetry assessment; no inference about hidden state |
| Game scripts/assets | `examples/artifacts/job.json` | Authoring spec | Accepted JSON artifact manifest for explicit safe local materialisation |

Image files are listed in the dataset's `images` array as relative paths beneath the dataset directory; they are encoded as inline image inputs, never sent as arbitrary remote URLs. See `include_images` on a reviewer to prevent unnecessary image-token expenditure. The artwork/NPC examples deliberately ship with `images:[]`: **their toy fixtures are not visual test results**. Add real screenshots and measured telemetry before any meaningful visual assessment.

Run the artifact example only after checking model billing:

```bash
quota-farm plan examples/artifacts/job.json
quota-farm run examples/artifacts/job.json --confirm-incentive
quota-farm materialize examples/artifacts/runs/game-artifact-RUN-ID \
    --output ./candidate-assets --confirm-write
```

Materialisation accepts only *unanimously accepted* candidate JSON manifests of the form:

```json
{
  "files": [
    {"path": "dialogue/start.txt", "encoding": "utf8", "content": "Original dialogue"},
    {"path": "data/asset.bin", "encoding": "base64", "content": "AAEC"}
  ]
}
```

It verifies relative portable paths, blocks traversal, duplicates, symlink parents, malformed base64, oversize outputs and existing files. It does **not** execute the content or integrate it into an authoritative repo. Prefer a deterministic offline asset compiler for large binaries over token-expensive LLM-generated base64.

## Supervisor integration

A supervisor invokes `quota-farm run JOB.json ...` as **one local command**, then reads the `summary.json` and individual `items/ID.json` receipts or exports accepted text to JSONL. The dispatcher makes independent requests in parallel without routing each token back through the supervisor. The supervisor alone is responsible for integration, actual compilation, test execution, repository writes, PRs and merge authorisation.

Use separate jobs for different projects. Keep original protected/held-out Sprocket/Neurounner data out of generation prompts, and distinguish synthetic records from actual model observations. To evaluate NPCs, extract frames and telemetry with your own deterministic game harness; this package does not claim to launch CyberSand.

## Accounting and operational safety

- SQLite stores cross-process reservation records in `$QUOTA_FARM_STATE_DIR/ledger.sqlite3` (default `~/.local/state/quota-farm`). Two pools use **UTC-day resets**. Paid requests use a configurable monthly start day (1–28) and separate per-run USD ceiling.
- Free-expected requests **refuse to execute** until the operator passes `--confirm-incentive` after verifying their own API project's data-sharing incentive tier. This flag is an acknowledgment, not proof; eligibility and tool/vision exclusions are determined by OpenAI.
- Missing usage or ambiguous network failures **retain full reserved usage** rather than being optimistically refunded. Actual usage is used for settlement when supplied. Provider retries default to **one** because a timed-out request may already have been charged.
- The local ledger is a conservative **scheduler**, not a vendor billing hard cap. Unexpected vendor charges, hidden input/image tokenisation, duplicate/retried requests, service-tier surcharges, other API projects and concurrent spend outside this CLI cannot be prevented by the local ledger. Use provider-side budget controls and inspect the actual Usage Dashboard.
- The Chat Completions wire payload contains only `model`, `messages`, `stream:false`, an output limit and (for OpenAI) `store:false`. No tool definitions or tool calls. Text content may still be billed and data sharing requires affirmative operator consent.
- Input specs, prompt templates and receipts live on disk. Preserve/secure them accordingly; don't commit secrets, private materials or original evidence to public repositories. Results have SHA-256 integrity checks but **not cryptographic signatures or independent evidence custody**.
- An accepted score is only the reviewers' opinion. Static models cannot prove level reachability, scientific truth, NPC frame-time, or gameplay correctness. Require deterministic tests and independent source-matched qualification.

## Tests

```bash
python -m unittest discover -s tests -v
```

CI tests four OS/Python combinations using mocked HTTP transport; no real API keys or spend. Full paid/free-incentive qualification and real graphical frame evaluations require separately authorised live pilot runs.

## Limitations and planned next steps

This v0.1 provides per-run batch dispatch, multiple independent reviewers, finite concurrency, local quotas, conservative paid accounting, offline tests, safe accepted-data export and explicit file materialisation. It intentionally does **not** include remote orchestration service, daemon, timed auto-spend, indefinite scheduling, resume-after-interruption, provider-side charge enforcement, automated training, automatic source writes, image generation, video decode or hosted tool-calling agents. External supervisors can use those capabilities while treating worker output as untrusted candidate data.
