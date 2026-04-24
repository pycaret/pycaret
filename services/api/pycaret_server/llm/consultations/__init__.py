"""One file per consultation type. Each file owns:

- its SYSTEM + user-prompt builder
- the JSON output schema sent to the provider
- a parser that validates the raw dict into `LLMAdvice`

Adding a new consultation type means dropping a new file here + hooking it
into `api/llm.py`.
"""
