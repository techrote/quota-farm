# Quota-farm agent development contract

- One model-call worker is **tool free**. Never add `tools`, `functions`, remote shell, code interpreter, MCP, or web tools to outbound worker requests.
- The coding supervisor owns repository writes and integrations; worker candidates never directly modify authoritative project state.
- Keep every accepted/rejected/failed worker result, reviewer disagreement, prompt hash, response hash and actual provider usage when supplied.
- Independently verify all free-incentive claims against provider billing. Configured pools are **not** proof of eligibility.
- Do not enable automatic paid fallbacks or interpret the local ledger as an enforceable upstream spending cap.
- Scripts/artifacts require deterministic local materialisation with explicit user/operator intent and path confinement. Never execute generated files automatically.
- Synthetic training examples must not be confused with empirical observations. Preserve held-outs and dataset provenance.
- Use GitHub-hosted Actions for offline tests; don't put live API keys in CI. Do not merge unqualified changes.
