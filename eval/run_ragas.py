"""Judge the saved agent runs with RAGAS metrics on the Gemini free tier.

Paced, call-capped and resumable: finished (question, mode, metric) scores are skipped, so a run that
stops on a quota error continues where it left off when rerun. Cheapest metrics run first.
Usage: python -m eval.run_ragas [--metrics NAME ...] [--max-calls 150] [--rpm 8] [--limit N]
"""
import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # ragas sends usage analytics unless this is set

import argparse
import asyncio
import json
import math
import time
import uuid
from collections import defaultdict
from pathlib import Path

import instructor
from google import genai
from google.genai import errors
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness

from agent.graph import FALLBACK_ANSWER
from llm.client import GEMINI_MODEL, LOG_PATH

EVAL_DIR = Path(__file__).resolve().parent
RUNS = EVAL_DIR / "results" / "agent_runs.jsonl"
SCORES = EVAL_DIR / "results" / "ragas_scores.jsonl"
MODES = ["dense", "hybrid", "hybrid_rerank"]
RATE_LIMIT_WAIT_S = 30
RATE_LIMIT_RETRIES = 3
REQUEST_TIMEOUT_S = 90
SERVER_ERROR_CODES = (500, 503)
SERVER_ERROR_WAIT_S = 10

METRICS = {  # cheapest first
    "context_recall": (ContextRecall, lambda r: dict(user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["reference"])),
    "factual_correctness": (FactualCorrectness, lambda r: dict(response=r["answer"], reference=r["reference"])),
    "faithfulness": (Faithfulness, lambda r: dict(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"])),
    "context_precision": (ContextPrecision, lambda r: dict(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"])),
}


class CallBudgetExceeded(Exception):
    pass


class QuotaExceeded(Exception):
    pass


class Pacer:
    """Spaces Gemini calls, enforces the call cap, handles 429s and logs every call."""

    def __init__(self, rpm: int, max_calls: int):
        self.interval = 60 / rpm
        self.max_calls = max_calls
        self.calls = 0
        self.last = 0.0
        self.metric = ""
        self.lock = None
        self.stop: Exception | None = None  # set when the run must end; survives exception wrapping

    async def wait(self):
        self.lock = self.lock or asyncio.Lock()
        async with self.lock:
            if self.calls >= self.max_calls:
                self.stop = CallBudgetExceeded(f"reached --max-calls {self.max_calls}")
                raise self.stop
            delay = self.last + self.interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.last = time.monotonic()
            self.calls += 1

    def log(self, started: float, response=None, error: str | None = None):
        usage = getattr(response, "usage_metadata", None)
        tokens_out = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        row = {
            "request_id": uuid.uuid4().hex,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": "gemini",
            "model": GEMINI_MODEL,
            "prompt": f"ragas.{self.metric}",
            "prompt_version": "ragas-0.4.3",
            "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
            "output_tokens": tokens_out,
            "latency_s": round(time.time() - started, 3),
            "cost_usd": 0.0,  # free tier
            "error": error,
        }
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")


def guard(client: genai.Client, pacer: Pacer) -> None:
    """Route the client's sync and async generate_content calls through the pacer."""
    sync_models, async_models = client.models, client.aio.models
    async_original = async_models.generate_content

    def quota_error(exc: errors.APIError) -> QuotaExceeded | None:
        message = str(exc)
        return QuotaExceeded(message[:200]) if "PerDay" in message or "per day" in message.lower() else None

    async def async_call(*args, **kwargs):
        for attempt in range(1, RATE_LIMIT_RETRIES + 1):
            await pacer.wait()
            started = time.time()
            try:
                response = await asyncio.wait_for(async_original(*args, **kwargs), timeout=REQUEST_TIMEOUT_S)
            except asyncio.TimeoutError:
                pacer.log(started, error=f"timeout after {REQUEST_TIMEOUT_S}s")
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                continue
            except errors.APIError as exc:
                pacer.log(started, error=f"{exc.code}: {str(exc)[:120]}")
                if exc.code in SERVER_ERROR_CODES and attempt < RATE_LIMIT_RETRIES:
                    await asyncio.sleep(SERVER_ERROR_WAIT_S * attempt)
                    continue
                if exc.code != 429:
                    raise
                if daily := quota_error(exc):
                    pacer.stop = daily
                    raise daily from exc
                await asyncio.sleep(RATE_LIMIT_WAIT_S * attempt)
                continue
            pacer.log(started, response)
            return response
        pacer.stop = QuotaExceeded(f"rate limit persisted after {RATE_LIMIT_RETRIES} retries")
        raise pacer.stop

    def sync_call(*args, **kwargs):
        raise RuntimeError("synchronous Gemini call is not paced or capped; ragas should use the async path")

    sync_models.generate_content = sync_call
    async_models.generate_content = async_call


async def score(metric, inputs: dict, pacer: Pacer):
    """Score one sample; if the run must stop, re-raise that reason however deeply it was wrapped."""
    try:
        return (await metric.ascore(**inputs)).value
    except Exception:
        if pacer.stop:
            raise pacer.stop
        raise


def finished_scores() -> dict[tuple[str, str, str], dict]:
    if not SCORES.exists():
        return {}
    rows = [json.loads(line) for line in SCORES.read_text().splitlines()]
    return {(r["id"], r["mode"], r["metric"]): r for r in rows if not (r["note"] or "").startswith("error")}


def summarize() -> None:
    by = defaultdict(list)
    for r in finished_scores().values():
        if r["value"] is not None:
            by[(r["metric"], r["mode"])].append(r["value"])
    print(f"\n{'metric':22s}" + "".join(f"{m:>22s}" for m in MODES))
    for metric in METRICS:
        cells = []
        for mode in MODES:
            values = by[(metric, mode)]
            cells.append(f"{sum(values) / len(values):.3f} (n={len(values)})" if values else "-")
        print(f"{metric:22s}" + "".join(f"{c:>22s}" for c in cells))


async def run(args) -> None:
    rows = [json.loads(line) for line in RUNS.read_text().splitlines()][: args.limit]
    pacer = Pacer(args.rpm, args.max_calls)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    guard(client, pacer)
    # ragas' llm_factory builds a sync instructor client for Google, but its metrics need the async one
    llm = InstructorLLM(client=instructor.from_genai(client, use_async=True), model=GEMINI_MODEL, provider="google", model_args=InstructorModelArgs())
    done = finished_scores()
    with SCORES.open("a") as out:
        for name in args.metrics:
            metric_cls, to_inputs = METRICS[name]
            metric = metric_cls(llm=llm)
            pacer.metric = name
            for r in rows:
                if (r["id"], r["mode"], name) in done:
                    continue
                fallback = r["answer"] == FALLBACK_ANSWER
                value, note = None, None
                if fallback and name == "factual_correctness":
                    value, note = 0.0, "fallback answer asserts nothing"
                elif fallback and name == "faithfulness":
                    note = "fallback answer, not scored"
                else:
                    try:
                        value = await score(metric, to_inputs(r), pacer)
                        value = None if value is None or math.isnan(value) else float(value)
                    except (CallBudgetExceeded, QuotaExceeded):
                        raise
                    except Exception as exc:  # one bad sample must not stop the run
                        note = f"error: {type(exc).__name__}: {str(exc)[:150]}"
                out.write(json.dumps({"id": r["id"], "mode": r["mode"], "metric": name, "value": value, "note": note}) + "\n")
                out.flush()
    print(f"all requested scores are complete ({pacer.calls} Gemini calls this run)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS))
    parser.add_argument("--max-calls", type=int, default=150)
    parser.add_argument("--rpm", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N saved runs")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (CallBudgetExceeded, QuotaExceeded) as exc:
        print(f"stopped: {type(exc).__name__}: {exc}\nrerun the same command later to resume")
    summarize()


if __name__ == "__main__":
    main()
