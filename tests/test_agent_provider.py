from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from searchpilot.agent.provider import MAX_RESPONSE_BYTES, HttpDecisionModel, ProviderError

MESSAGES = [{"role": "user", "content": "private question"}]


def envelope(content: str = '{"action":"finish"}', **updates: Any) -> dict[str, Any]:
    return {
        "model": "reported-model",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        **updates,
    }


def provider(handler: Any, **kwargs: Any) -> HttpDecisionModel:
    return HttpDecisionModel(
        "https://provider.example/v1/chat/completions",
        "configured-model",
        "private-api-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_success_request_contract_metadata_and_budget() -> None:
    requests: list[httpx.Request] = []
    events: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert len(events) == 1 and events[0]["status"] == "pending"
        assert events[0]["usage"] is None
        requests.append(request)
        return httpx.Response(
            200,
            json=envelope(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        )

    async with provider(handler, max_requests=1, event_sink=events.append) as model:
        assert await model.decide(MESSAGES) == {"action": "finish"}
        with pytest.raises(ProviderError, match="^budget_exhausted$"):
            await model.decide(MESSAGES)
        summary = model.summary()
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer private-api-key"
    body = json.loads(requests[0].content)
    assert body == {
        "model": "configured-model",
        "messages": MESSAGES,
        "temperature": 0,
        "max_tokens": 2048,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[1]["response_model"] is None
    assert events[1]["response_model_matches_configured"] is False
    assert len(events[1]["response_model_sha256"]) == 64
    assert summary["requests_sent"] == summary["requests_reserved"] == 1
    assert summary["requests_completed"] == summary["usage_known_requests"] == 1
    assert summary["total_tokens"] == 15
    assert summary["output_tokens_reserved"] == summary["max_output_tokens"] == 2048
    safe = json.dumps(events)
    assert all(
        secret not in safe for secret in ["provider.example", "private-api-key", "private question"]
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/chat",
        "https://user:password@example.com/chat",
        "https://user@example.com/chat",
        "https://example.com/chat?",
        "https://example.com/chat?key=secret",
        "https://example.com/chat#",
        "https://example.com\\@localhost/chat",
        "https://example.com/\nchat",
        "https://example.com:99999/chat",
        "https://example.com:0/chat",
        "http://127.1/chat",
        "http://localhost.evil/chat",
        "http://%6cocalhost/chat",
        "file:///chat",
        "https:///chat",
    ],
)
def test_reject_dangerous_url(endpoint: str) -> None:
    with pytest.raises(ProviderError, match="^invalid_endpoint$"):
        HttpDecisionModel(endpoint, "fake")


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com/chat",
        "http://localhost:123/chat",
        "http://127.0.0.1/chat",
        "http://[::1]/chat",
    ],
)
def test_allowed_url(endpoint: str) -> None:
    HttpDecisionModel(endpoint, "fake")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_requests": 9},
        {"max_requests": True},
        {"max_tokens_per_request": 2049},
        {"request_timeout_seconds": float("nan")},
        {"request_timeout_seconds": True},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": 301},
        {"api_key": "secret\r\nheader: value"},
        {"model": "invalid model"},
    ],
)
def test_configuration_bounds(kwargs: dict[str, Any]) -> None:
    args = {"model": "fake", **kwargs}
    with pytest.raises(ProviderError, match="^invalid_configuration$"):
        HttpDecisionModel("https://example.com/chat", **args)


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"choices":[],"choices":[]}',
        '{"value": NaN}',
        '{"value": 1e9999}',
        '{"choices":[true]}',
        '{"choices":null}',
        json.dumps(envelope('{"n": NaN}')),
        json.dumps(envelope('{"n": Infinity}')),
        json.dumps(envelope('{"n": 1e999}')),
        json.dumps(envelope('{"nested":{"a":1,"a":2}}')),
        json.dumps(envelope("[]")),
        "\ud800",
    ],
)
async def test_invalid_envelope_or_content(raw: str) -> None:
    async with provider(
        lambda _: httpx.Response(200, content=raw.encode("utf-8", errors="surrogatepass"))
    ) as model:
        with pytest.raises(ProviderError, match="^invalid_response$"):
            await model.decide(MESSAGES)
        assert model.summary()["requests_failed"] == 1


@pytest.mark.parametrize(
    "finish", ["length", "content_filter", "tool_calls", "function_call", None, "secret"]
)
async def test_abnormal_finish_rejected_and_sanitized(finish: Any) -> None:
    value = envelope()
    value["choices"][0]["finish_reason"] = finish
    async with provider(lambda _: httpx.Response(200, json=value)) as model:
        with pytest.raises(ProviderError, match="^incomplete_response$"):
            await model.decide(MESSAGES)
        assert "secret" not in json.dumps(model.events)


@pytest.mark.parametrize("field", ["tool_calls", "function_call", "refusal"])
async def test_native_tools_and_refusal_rejected(field: str) -> None:
    value = envelope()
    value["choices"][0]["message"][field] = {"private": "data"}
    async with provider(lambda _: httpx.Response(200, json=value)) as model:
        with pytest.raises(ProviderError, match="^incomplete_response$"):
            await model.decide(MESSAGES)


async def test_redirect_has_no_followup_and_unknown_usage() -> None:
    count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(302, headers={"Location": "https://evil.example/private"})

    async with provider(handler) as model:
        with pytest.raises(ProviderError, match="^redirect_rejected$"):
            await model.decide(MESSAGES)
        assert model.summary()["usage_unknown_requests"] == 1
        assert model.summary()["total_tokens"] is None
    assert count == 1


async def test_timeout_covers_entire_stream_and_closes_it() -> None:
    closed = False

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(20):
                await asyncio.sleep(0.01)
                yield b" "

        async def aclose(self):
            nonlocal closed
            closed = True

    async with provider(
        lambda _: httpx.Response(200, stream=SlowStream()), request_timeout_seconds=0.04
    ) as model:
        with pytest.raises(ProviderError, match="^timeout$"):
            await model.decide(MESSAGES)
        assert model.events[-1]["status"] == "timeout"
        assert model.summary()["requests_sent"] == 1
    assert closed


async def test_oversize_response_is_capped() -> None:
    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(MAX_RESPONSE_BYTES // 65536 + 1):
                yield b"x" * 65536

    async with provider(lambda _: httpx.Response(200, stream=LargeStream())) as model:
        with pytest.raises(ProviderError, match="^response_too_large$"):
            await model.decide(MESSAGES)
        assert model.summary()["usage_unknown_requests"] == 1


async def test_cancellation_propagates_and_remains_unknown() -> None:
    entered = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async with provider(handler) as model:
        task = asyncio.create_task(model.decide(MESSAGES))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.events[-1]["status"] == "cancelled"
        assert model.summary()["requests_completed"] == 0
        assert model.summary()["usage_unknown_requests"] == 1


@pytest.mark.parametrize(
    "usage", [None, {}, {"completion_tokens": 3}, {"total_tokens": True}, {"total_tokens": -1}]
)
async def test_usage_unknown_is_never_zero_usage(usage: Any) -> None:
    async with provider(lambda _: httpx.Response(200, json=envelope(usage=usage))) as model:
        await model.decide(MESSAGES)
        assert model.summary()["total_tokens"] is None
        assert model.summary()["usage_known_requests"] == 0
        assert model.summary()["usage_unknown_requests"] == 1


async def test_safe_transport_error_and_response_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "private-api-key provider.example private question", request=request
        )

    async with provider(handler) as model:
        with pytest.raises(ProviderError, match="^http_error$"):
            await model.decide(MESSAGES)
        assert "private" not in json.dumps(model.events)
    async with provider(
        lambda _: httpx.Response(200, json=envelope(model="private-api-key"))
    ) as model:
        await model.decide(MESSAGES)
        assert model.events[-1]["response_model"] is None


async def test_sink_failure_before_send_and_after_result() -> None:
    count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json=envelope())

    def broken_sink(event: dict[str, Any]) -> None:
        raise RuntimeError("private sink details")

    async with provider(handler, event_sink=broken_sink) as model:
        with pytest.raises(ProviderError, match="^event_sink_error$"):
            await model.decide(MESSAGES)
        assert model.summary()["requests_sent"] == count == 0

    def result_broken_sink(event: dict[str, Any]) -> None:
        if event["event"] == "request_result":
            raise RuntimeError("private sink details")

    async with provider(handler, event_sink=result_broken_sink) as model:
        with pytest.raises(ProviderError, match="^event_sink_error$"):
            await model.decide(MESSAGES)
        assert model.summary()["requests_completed"] == 0


async def test_events_are_defensive_copies_and_request_ids_unique() -> None:
    async with provider(lambda _: httpx.Response(200, json=envelope())) as model:
        await model.decide(MESSAGES)
        await model.decide(MESSAGES)
        events = model.events
        assert events[0]["request_id"] != events[2]["request_id"]
        events[0]["status"] = "changed"
        assert model.events[0]["status"] == "pending"


async def test_mixed_usage_preserves_partial_known_total() -> None:
    count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        usage = {"total_tokens": 12} if count == 1 else {"completion_tokens": 3}
        return httpx.Response(200, json=envelope(usage=usage))

    async with provider(handler) as model:
        await model.decide(MESSAGES)
        await model.decide(MESSAGES)
        summary = model.summary()
        assert summary["known_total_tokens"] == 12
        assert summary["total_tokens"] is None
        assert summary["usage_known_requests"] == summary["usage_unknown_requests"] == 1


@pytest.mark.parametrize(
    "reported",
    ["https://provider.example/v1/chat/completions", "private-question", "configured-model"],
)
async def test_reported_model_cannot_write_arbitrary_text_to_ledger(reported: str) -> None:
    async with provider(lambda _: httpx.Response(200, json=envelope(model=reported))) as model:
        await model.decide(MESSAGES)
        result = model.events[-1]
        assert result["response_model_matches_configured"] == (reported == "configured-model")
        if reported == "configured-model":
            assert result["response_model"] == "configured-model"
        else:
            assert reported not in json.dumps(model.events)


async def test_failed_request_reservation_is_not_refunded() -> None:
    count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(503, text="private error body")

    async with provider(handler, max_requests=1) as model:
        with pytest.raises(ProviderError, match="^http_error$"):
            await model.decide(MESSAGES)
        with pytest.raises(ProviderError, match="^budget_exhausted$"):
            await model.decide(MESSAGES)
        assert model.summary()["requests_sent"] == count == 1
        assert model.summary()["output_tokens_reserved"] == 2048
        assert "private error body" not in json.dumps(model.events)


async def test_not_open_and_invalid_messages_send_nothing() -> None:
    model = provider(lambda _: httpx.Response(200, json=envelope()))
    with pytest.raises(ProviderError, match="^provider_not_open$"):
        await model.decide(MESSAGES)
    async with model:
        with pytest.raises(ProviderError, match="^invalid_messages$"):
            await model.decide([{"role": "tool", "content": "bad"}])
        with pytest.raises(ProviderError, match="^invalid_messages$"):
            await model.decide([{"role": [], "content": "bad"}])
        assert model.summary()["requests_sent"] == 0


@pytest.mark.parametrize(
    "body,phase,reason",
    [
        (b"not JSON private-api-key", "envelope_json", "json_syntax"),
        (b'{"choices":[],"choices":[]}', "envelope_json", "duplicate_key"),
        (b'{"n":NaN}', "envelope_json", "non_finite_number"),
        (b"[]", "envelope_json", "object_required"),
        (b"\xff", "envelope_json", "invalid_utf8"),
        (b'{"choices":[]}', "envelope_json", "invalid_choices"),
        (
            json.dumps(envelope("private-api-key invalid JSON")).encode(),
            "content_json",
            "json_syntax",
        ),
        (json.dumps(envelope('{"n":1,"n":2}')).encode(), "content_json", "duplicate_key"),
        (json.dumps(envelope('{"n":1e9999}')).encode(), "content_json", "non_finite_number"),
    ],
)
async def test_parse_diagnostics_are_fixed_phase_and_reason_without_raw_text(
    body: bytes,
    phase: str,
    reason: str,
) -> None:
    async with provider(lambda _: httpx.Response(200, content=body)) as model:
        with pytest.raises(ProviderError, match="^invalid_response$"):
            await model.decide(MESSAGES)
        event = model.events[-1]
        assert event["validation_phase"] == phase
        assert event["validation_reason"] == reason
        safe = json.dumps(model.events)
        assert "private-api-key" not in safe and "private question" not in safe
