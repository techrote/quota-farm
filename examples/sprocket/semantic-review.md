Independently assess synthetic instruction/response semantic consistency, absence of unsupported scientific claims, and training-usefulness. Distinguish synthetic examples from observed model behaviour. Do not claim to have run a model or inspected held-outs.

Task: ${input}
Context: ${context}
Candidate: ${candidate}

Return JSON only: {"score":0.0,"verdict":"accept|reject","findings":["specific evidence"]}. Be stringent. Use reject if uncertain.
