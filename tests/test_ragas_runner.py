import asyncio

import pytest

from eval.run_ragas import CallBudgetExceeded, Pacer, QuotaExceeded, score


class FailingMetric:
    """Stands in for a ragas metric whose llm call failed with a wrapped, generic error."""

    def __init__(self, error):
        self.error = error

    async def ascore(self, **inputs):
        raise self.error


def test_pacer_stops_at_the_call_cap_and_remembers_why():
    async def run():
        pacer = Pacer(rpm=6000, max_calls=2)
        await pacer.wait()
        await pacer.wait()
        with pytest.raises(CallBudgetExceeded):
            await pacer.wait()
        return pacer

    pacer = asyncio.run(run())
    assert pacer.calls == 2 and isinstance(pacer.stop, CallBudgetExceeded)


def test_score_reraises_the_stop_reason_even_when_a_retry_layer_wrapped_it():
    pacer = Pacer(rpm=60, max_calls=10)
    pacer.stop = QuotaExceeded("daily quota")
    wrapped = RuntimeError("InstructorRetryException: all attempts failed")
    with pytest.raises(QuotaExceeded):
        asyncio.run(score(FailingMetric(wrapped), {}, pacer))


def test_score_lets_ordinary_errors_through_when_the_run_is_not_stopping():
    pacer = Pacer(rpm=60, max_calls=10)
    with pytest.raises(ValueError):
        asyncio.run(score(FailingMetric(ValueError("bad sample")), {}, pacer))
