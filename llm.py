"""Phase 4 — the Ollama connection every agent goes through.

Two models are available locally and they are not interchangeable:
  qwen2.5-coder — reading/reasoning about code (Mapper, Critic)
  qwen2.5       — writing prose for humans (Writer)

IMPORTANT (measured on this machine): only one 5GB model fits in RAM at a time,
so every switch between them evicts the other and reloads from disk:

    same model again ...  ~2.9s
    switched model .....  ~15s

Phase 8 must therefore **batch by model** — run every Mapper call, then every
Writer call, then every Critic call — instead of cycling per symbol. Cycling
three models per symbol costs ~42s of pure loading each, which is minutes of
dead time on any real repo. Set LEGIBLE_SINGLE_MODEL=1 to route prose through the
code model too and avoid swaps entirely.

Run `python llm.py` for a health check + smoke test.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import ollama

CODE_MODEL = os.getenv("LEGIBLE_CODE_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
PROSE_MODEL = os.getenv("LEGIBLE_PROSE_MODEL", "qwen2.5:7b-instruct-q4_K_M")

# One model for everything — trades some prose quality for zero swap overhead.
if os.getenv("LEGIBLE_SINGLE_MODEL") == "1":
    PROSE_MODEL = CODE_MODEL

# Ollama unloads after 5 minutes by default. A pipeline that pauses to parse or
# embed between calls would come back to a cold model and eat the reload.
KEEP_ALIVE = os.getenv("LEGIBLE_KEEP_ALIVE", "30m")

# Ollama defaults to a 4096-token context. Code prompts blow past that and the
# overflow is dropped *silently* from the front — which would quietly delete the
# instructions. Raise it; qwen2.5 handles 32k, but bigger costs CPU RAM.
NUM_CTX = int(os.getenv("LEGIBLE_NUM_CTX", "8192"))

DEFAULT_TIMEOUT = float(os.getenv("LEGIBLE_TIMEOUT", "300"))  # 7B on CPU is slow
MAX_RETRIES = 2


class LLMError(RuntimeError):
    """Ollama unreachable, model missing, or output unusable after retries."""


@dataclass
class Reply:
    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1

    @property
    def tokens_per_sec(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0

    def __str__(self) -> str:
        return self.text


@dataclass
class Stats:
    """Cumulative cost tracker — CPU inference is the bottleneck of this project."""

    calls: int = 0
    seconds: float = 0.0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    swaps: int = 0
    _last_model: str | None = None

    def record(self, r: Reply) -> None:
        self.calls += 1
        self.seconds += r.seconds
        self.output_tokens += r.output_tokens
        self.by_model[r.model] = self.by_model.get(r.model, 0) + 1
        if self._last_model is not None and r.model != self._last_model:
            self.swaps += 1
        self._last_model = r.model

    def summary(self) -> str:
        if not self.calls:
            return "[llm] no calls yet"
        line = (f"[llm] {self.calls} calls, {self.seconds:.1f}s total "
                f"({self.seconds / self.calls:.1f}s avg), "
                f"{self.output_tokens} output tokens, models={self.by_model}")
        if self.swaps:
            cost = f"{self.swaps} model swaps (~{self.swaps * 12}s of reload)"
            # A few swaps are unavoidable: each pipeline round legitimately hands
            # off prose model -> code model and back. Only flag genuine thrashing,
            # or the warning cries wolf on a correctly batched run.
            if self.swaps * 4 > self.calls:
                line += (f"\n[llm] WARNING {cost} across only {self.calls} calls — "
                         f"batch by model, see the note in llm.py.")
            else:
                line += f"\n[llm] {cost}, expected for multi-round batching."
        return line


STATS = Stats()


def _client(timeout: float) -> ollama.Client:
    return ollama.Client(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        timeout=timeout,
    )


def available_models() -> list[str]:
    try:
        resp = _client(15).list()
    except Exception as exc:
        raise LLMError(
            f"Cannot reach Ollama at {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}. "
            f"Is it running? Start it with `ollama serve`.\n  {exc}"
        ) from exc
    out = []
    for m in resp.get("models", []):
        name = m.get("model") or m.get("name")
        if name:
            out.append(name)
    return out


def ensure_ready(model: str = CODE_MODEL) -> None:
    """Fail loudly *now* rather than three phases deep into an agent loop."""
    models = available_models()
    if model in models:
        return
    # Ollama treats a bare name as ":latest"; accept an exact-prefix match too.
    if any(m.split(":")[0] == model.split(":")[0] for m in models):
        near = [m for m in models if m.split(":")[0] == model.split(":")[0]]
        raise LLMError(
            f"Model {model!r} not found, but these look related: {near}\n"
            f"Set LEGIBLE_CODE_MODEL to one of them, or run: ollama pull {model}"
        )
    raise LLMError(
        f"Model {model!r} not installed. Available: {models or '(none)'}\n"
        f"Run: ollama pull {model}"
    )


def chat(
    prompt: str,
    system: str | None = None,
    model: str = CODE_MODEL,
    temperature: float = 0.1,
    json_schema: dict[str, Any] | None = None,
    force_json: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    num_ctx: int = NUM_CTX,
) -> Reply:
    """One call to Ollama. Low temperature by default — this is analysis, not fiction."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    fmt: Any = None
    if json_schema is not None:
        fmt = json_schema  # structured output — the strongest guarantee
    elif force_json:
        fmt = "json"

    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        started = time.monotonic()
        try:
            resp = _client(timeout).chat(
                model=model,
                messages=messages,
                format=fmt,
                keep_alive=KEEP_ALIVE,
                options={"temperature": temperature, "num_ctx": num_ctx},
            )
        except Exception as exc:
            last = exc
            if attempt <= MAX_RETRIES:
                print(f"[llm] attempt {attempt} failed ({exc}); retrying")
                continue
            raise LLMError(f"{model} failed after {attempt} attempts: {exc}") from exc

        reply = Reply(
            text=resp["message"]["content"].strip(),
            model=model,
            seconds=time.monotonic() - started,
            prompt_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            attempts=attempt,
        )
        STATS.record(reply)
        return reply

    raise LLMError(str(last))


def _extract_json(text: str) -> str:
    """Salvage JSON from a reply wrapped in prose or ``` fences."""
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            body = part.split("\n", 1)[-1] if part[:20].strip().lower().startswith("json") else part
            if body.strip().startswith(("{", "[")):
                return body.strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            return text[start : end + 1]
    return text


def chat_json(
    prompt: str,
    system: str | None = None,
    model: str = CODE_MODEL,
    schema: dict[str, Any] | None = None,
    **kw: Any,
) -> tuple[Any, Reply]:
    """Chat that must return JSON. Phases 5 and 7 depend on this being reliable.

    Passing a `schema` uses Ollama structured outputs, which constrains decoding
    so the result is valid by construction. Without one it falls back to JSON
    mode plus a salvage pass.
    """
    reply = chat(prompt, system=system, model=model,
                 json_schema=schema, force_json=schema is None, **kw)
    try:
        return json.loads(reply.text), reply
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_extract_json(reply.text)), reply
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"{model} did not return valid JSON: {exc}\n--- raw ---\n{reply.text[:600]}"
        ) from exc


def _smoke() -> int:
    print(f"[llm] code model:  {CODE_MODEL}")
    print(f"[llm] prose model: {PROSE_MODEL}")
    print(f"[llm] num_ctx={NUM_CTX} timeout={DEFAULT_TIMEOUT}s\n")

    try:
        models = available_models()
    except LLMError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"[llm] installed: {models}")

    failures = []
    for label, m in (("code", CODE_MODEL), ("prose", PROSE_MODEL)):
        try:
            ensure_ready(m)
            print(f"[llm] {label} model OK")
        except LLMError as exc:
            failures.append(f"{label}: {exc}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1

    print("\n--- 1. plain generation ---")
    r = chat("Reply with exactly the word: ready", temperature=0.0)
    print(f"  {r.text!r}  ({r.seconds:.1f}s, {r.output_tokens} tok, "
          f"{r.tokens_per_sec:.1f} tok/s)")

    print("\n--- 2. structured output (what Phases 5 & 7 rely on) ---")
    schema = {
        "type": "object",
        "properties": {
            "purpose": {"type": "string"},
            "calls": {"type": "array", "items": {"type": "string"}},
            "is_entry_point": {"type": "boolean"},
        },
        "required": ["purpose", "calls", "is_entry_point"],
    }
    code = (
        "def wsgi_app(self, environ, start_response):\n"
        "    ctx = self.request_context(environ)\n"
        "    ctx.push()\n"
        "    rv = self.full_dispatch_request()\n"
        "    return rv(environ, start_response)\n"
    )
    data, r2 = chat_json(
        f"Analyse this function.\n\n{code}\n\n"
        "purpose: one sentence. calls: functions it calls. "
        "is_entry_point: true if external code calls it.",
        system="You are a precise code analyst. Answer only from the code shown.",
        schema=schema,
    )
    print(f"  {json.dumps(data, indent=2)[:400]}")
    print(f"  ({r2.seconds:.1f}s, {r2.output_tokens} tok)")

    ok = isinstance(data.get("calls"), list) and isinstance(data.get("purpose"), str)
    print(f"  schema honoured: {ok}")
    if not ok:
        return 1

    print("\n--- 3. prose model ---")
    r3 = chat("In one short sentence, what is a web framework?",
              model=PROSE_MODEL, temperature=0.3)
    print(f"  {r3.text[:200]}")
    print(f"  ({r3.seconds:.1f}s, {r3.tokens_per_sec:.1f} tok/s)")

    print(f"\n{STATS.summary()}")
    print("\nOK - Ollama reachable, both models work, structured output verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
