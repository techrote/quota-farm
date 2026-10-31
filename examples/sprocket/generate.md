Generate exactly one *synthetic* Sprocket training example, as a JSON object with keys "instruction", "response", "rationale", "synthetic":true and "source_id". The "source_id" must equal ${id}. Do not invent observations or outcomes; never reproduce or infer held-out tests.

Task:
${input}

Context:
${context}

Output only valid JSON, without Markdown fences.
