Independently assess the actor telemetry for measured decision coherence, state transitions and measurable anomalies. Do not infer visual quality. Synthetic fixtures must never be treated as real observations.

Scenario: ${input}
Actor: ${candidate}
Telemetry: ${telemetry}

Return JSON only: {"score":0.0,"verdict":"accept|reject","findings":["measured anomalies or missing data"]}.
