"""One interface over local Ollama and the Gemini API.

Every call returns a pydantic-validated object and appends one row to logs/llm_calls.jsonl
(model, prompt name and version, tokens, latency, cost). Prompt text is never logged.
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import TypeVar

import requests
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
LOG_PATH = ROOT / "logs" / "llm_calls.jsonl"

OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # digest 0edcdef34593
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_USD_PER_M_TOKENS = {"input": 0.30, "output": 2.50}
OLLAMA_NUM_CTX = 4096  # Ollama's default of 2048 silently truncates RAG prompts; 4096 fits 5 passages and keeps VRAM use low
MAX_ATTEMPTS = 2

T = TypeVar("T", bound=BaseModel)

load_dotenv(ROOT / ".env")


def load_prompt(name: str, version: int) -> tuple[str, str]:
    """Return (system, user_template) from prompts/<name>.v<version>.md, split on the first '---' line."""
    text = (PROMPTS_DIR / f"{name}.v{version}.md").read_text()
    system, user = text.split("\n---\n", 1)
    return system.strip(), user.strip()


class LLM:
    def __init__(self, backend: str | None = None):
        self.backend = backend or os.getenv("LLM_BACKEND", "ollama")
        if self.backend not in ("ollama", "gemini"):
            raise ValueError(f"unknown backend {self.backend!r}, expected 'ollama' or 'gemini'")
        self.model = OLLAMA_MODEL if self.backend == "ollama" else GEMINI_MODEL

    def structured(self, prompt: str, version: int, schema: type[T], **variables: str) -> T:
        system, user_template = load_prompt(prompt, version)
        user = user_template.format(**variables)
        error = None
        for _ in range(MAX_ATTEMPTS):
            attempt_user = user if error is None else f"{user}\n\nYour previous answer was invalid ({error}). Reply with valid JSON only."
            started = time.time()
            text, tokens_in, tokens_out = self._call(system, attempt_user, schema)
            try:
                result = schema.model_validate_json(text)
                error = None
            except ValidationError as exc:
                error = exc.errors()[0]["msg"]
            self._log(prompt, version, tokens_in, tokens_out, time.time() - started, error)
            if error is None:
                return result
        raise ValueError(f"{prompt}.v{version}: no valid {schema.__name__} after {MAX_ATTEMPTS} attempts ({error})")

    @traceable(run_type="llm", name="llm_call", process_inputs=lambda inputs: {"system": inputs["system"], "user": inputs["user"]})
    def _call(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        return self._call_ollama(system, user, schema) if self.backend == "ollama" else self._call_gemini(system, user, schema)

    def _call_ollama(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": schema.model_json_schema(),
                "stream": False,
                "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        return body["message"]["content"], body.get("prompt_eval_count", 0), body.get("eval_count", 0)

    def _call_gemini(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            ),
        )
        usage = response.usage_metadata
        tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return response.text, usage.prompt_token_count or 0, tokens_out

    def _cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        if self.backend == "ollama":
            return 0.0
        return (tokens_in * GEMINI_USD_PER_M_TOKENS["input"] + tokens_out * GEMINI_USD_PER_M_TOKENS["output"]) / 1_000_000

    def _log(self, prompt: str, version: int, tokens_in: int, tokens_out: int, latency: float, error: str | None) -> None:
        row = {
            "request_id": uuid.uuid4().hex,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": self.backend,
            "model": self.model,
            "prompt": prompt,
            "prompt_version": version,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "latency_s": round(latency, 3),
            "cost_usd": round(self._cost_usd(tokens_in, tokens_out), 6),
            "error": error,
        }
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
