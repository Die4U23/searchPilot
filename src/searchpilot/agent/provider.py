"""Bounded OpenAI-compatible JSON decisions, without demo or evaluation imports.

One instance belongs to one run. Reservations are never refunded, including on
timeouts and cancellation; reported token usage is separate from these limits.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import ipaddress
import json
import math
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from searchpilot.agent.errors import DecisionError

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,127}\Z")
_ERROR_CODES = frozenset(
    {
        "invalid_configuration",
        "invalid_endpoint",
        "invalid_messages",
        "provider_not_open",
        "provider_already_open",
        "budget_exhausted",
        "timeout",
        "http_error",
        "authentication_failed",
        "rate_limited",
        "redirect_rejected",
        "response_too_large",
        "invalid_response",
        "incomplete_response",
        "transport_error",
        "event_sink_error",
    }
)


class ProviderError(DecisionError):
    """A fixed public code; never contains response, URL, or exception text."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERROR_CODES else "transport_error"
        super().__init__(self.code)


def _validate_endpoint(endpoint: str) -> str:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or any(char.isspace() or ord(char) < 32 for char in endpoint)
        or any(char in endpoint for char in "\\?#")
    ):
        raise ProviderError("invalid_endpoint")
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
        if (
            not hostname
            or parsed.scheme not in {"https", "http"}
            or parsed.username is not None
            or parsed.password is not None
            or "%" in parsed.netloc
            or port == 0
        ):
            raise ValueError
        if (
            parsed.scheme == "http"
            and hostname != "localhost"
            and not ipaddress.ip_address(hostname).is_loopback
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise ProviderError("invalid_endpoint") from None
    return endpoint


class _JsonProtocolError(ValueError):
    """Internal fixed JSON reason, never a JSON parser's raw error text."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _parse_object(raw: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise _JsonProtocolError("non_finite_number")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise _JsonProtocolError("non_finite_number")
        return result

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _JsonProtocolError("duplicate_key")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_keys,
        )
    except json.JSONDecodeError:
        raise _JsonProtocolError("json_syntax") from None
    if not isinstance(parsed, dict):
        raise _JsonProtocolError("object_required")
    return parsed


class HttpDecisionModel:
    """Async per-run provider; ``events`` and ``summary()`` contain safe metadata.

    ``event_sink`` is a synchronous callback. A pending event is emitted before
    sending. A sink failure prevents a successful return, and never causes a retry.
    ``requests_sent`` counts dispatch attempts, not proof that a server received
    them. Usage stays unknown until a provider reports a valid total_tokens field.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        *,
        max_requests: int = 8,
        max_tokens_per_request: int = 2048,
        request_timeout_seconds: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        if (
            not isinstance(model, str)
            or not _MODEL_NAME.fullmatch(model)
            or "://" in model
            or (
                api_key is not None
                and (not isinstance(api_key, str) or "\r" in api_key or "\n" in api_key)
            )
            or (api_key and api_key in model)
            or type(max_requests) is not int
            or not 1 <= max_requests <= 8
            or type(max_tokens_per_request) is not int
            or not 1 <= max_tokens_per_request <= 2048
            or type(request_timeout_seconds) not in {int, float}
            or not math.isfinite(request_timeout_seconds)
            or not 0 < request_timeout_seconds <= 300
            or (event_sink is not None and not callable(event_sink))
        ):
            raise ProviderError("invalid_configuration")
        self._model = model
        self._api_key = api_key
        self._max_requests = max_requests
        self._max_tokens = max_tokens_per_request
        self._timeout = request_timeout_seconds
        self._transport = transport
        self._sink = event_sink
        self._client: httpx.AsyncClient | None = None
        self._events: list[dict[str, Any]] = []
        self._requests_reserved = 0
        self._output_reserved = 0
        self._sent = 0

    async def __aenter__(self) -> HttpDecisionModel:
        if self._client is not None:
            raise ProviderError("provider_already_open")
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)

    def _emit(self, event: dict[str, Any]) -> None:
        self._events.append(copy.deepcopy(event))
        if self._sink is not None:
            try:
                self._sink(copy.deepcopy(event))
            except Exception:
                raise ProviderError("event_sink_error") from None

    def summary(self) -> dict[str, Any]:
        results = [event for event in self._events if event["event"] == "request_result"]
        sent_results = [event for event in results if event["sent"]]
        known = [event["usage"] for event in sent_results if event["usage_known"]]
        known_total = sum(usage["total_tokens"] for usage in known)
        return {
            "requests_reserved": self._requests_reserved,
            "requests_sent": self._sent,
            "requests_completed": sum(event["status"] == "completed" for event in results),
            "requests_failed": sum(event["status"] != "completed" for event in results),
            "output_tokens_reserved": self._output_reserved,
            "max_requests": self._max_requests,
            "max_output_tokens": self._max_requests * self._max_tokens,
            "max_tokens_per_request": self._max_tokens,
            "usage_known_requests": len(known),
            "usage_unknown_requests": self._sent - len(known),
            "known_total_tokens": known_total,
            "total_tokens": known_total if len(known) == self._sent else None,
            "retry_count": 0,
        }

    async def decide(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self._client is None:
            raise ProviderError("provider_not_open")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message["role"], str)
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)
                for message in messages
            )
        ):
            raise ProviderError("invalid_messages")
        if (
            self._requests_reserved >= self._max_requests
            or self._output_reserved + self._max_tokens > self._max_requests * self._max_tokens
        ):
            raise ProviderError("budget_exhausted")
        self._requests_reserved += 1
        self._output_reserved += self._max_tokens
        event: dict[str, Any] = {
            "event": "request",
            "request_id": "agent_" + uuid4().hex,
            "status": "pending",
            "sent": False,
            "usage": None,
            "usage_known": False,
            "response_model": None,
            "response_model_sha256": None,
            "response_model_matches_configured": None,
            "finish_reason": None,
            "validation_phase": None,
            "validation_reason": None,
            "configured_model": self._model,
            "reserved_output_tokens": self._max_tokens,
            "requests_reserved_total": self._requests_reserved,
            "output_tokens_reserved_total": self._output_reserved,
        }
        start = time.monotonic()
        try:
            self._emit(event)
            async with asyncio.timeout(self._timeout):
                self._sent += 1
                event["sent"] = True
                async with self._client.stream(
                    "POST",
                    self._endpoint,
                    headers={"X-Request-ID": event["request_id"]},
                    json={
                        "model": self._model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": self._max_tokens,
                        "stream": False,
                        "response_format": {"type": "json_object"},
                    },
                ) as response:
                    event["http_status"] = response.status_code
                    if 300 <= response.status_code < 400:
                        raise ProviderError("redirect_rejected")
                    if response.status_code in {401, 403}:
                        raise ProviderError("authentication_failed")
                    if response.status_code == 429:
                        raise ProviderError("rate_limited")
                    if not 200 <= response.status_code < 300:
                        raise ProviderError("http_error")
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise ProviderError("response_too_large")
                        body.extend(chunk)
                    event["validation_phase"] = "envelope_json"
                    raw = _parse_object(body.decode("utf-8"))
                self._read_metadata(raw, event)
                choices = raw.get("choices")
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                ):
                    event["validation_reason"] = "invalid_choices"
                    raise ProviderError("invalid_response")
                choice = choices[0]
                finish = choice.get("finish_reason")
                event["finish_reason"] = (
                    finish
                    if isinstance(finish, str)
                    and finish
                    in {"stop", "length", "content_filter", "tool_calls", "function_call"}
                    else "unknown"
                )
                message = choice.get("message")
                if (
                    finish != "stop"
                    or not isinstance(message, dict)
                    or message.get("tool_calls")
                    or message.get("function_call")
                    or message.get("refusal")
                    or not isinstance(message.get("content"), str)
                ):
                    event["validation_reason"] = "incomplete_or_unexpected_message"
                    raise ProviderError("incomplete_response")
                event["validation_phase"] = "content_json"
                result = _parse_object(message["content"])
                event["status"] = "completed"
                return result
        except asyncio.CancelledError:
            event["status"] = "cancelled"
            raise
        except (TimeoutError, httpx.TimeoutException):
            event["status"] = "timeout"
            raise ProviderError("timeout") from None
        except ProviderError as exc:
            event["status"] = exc.code
            raise
        except _JsonProtocolError as exc:
            event["status"] = "invalid_response"
            event["validation_reason"] = exc.reason
            raise ProviderError("invalid_response") from None
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            event["status"] = "invalid_response"
            event["validation_reason"] = (
                "invalid_utf8" if isinstance(exc, UnicodeError) else "json_type_or_limit"
            )
            raise ProviderError("invalid_response") from None
        except httpx.HTTPError:
            event["status"] = "http_error"
            raise ProviderError("http_error") from None
        except Exception:
            event["status"] = "transport_error"
            raise ProviderError("transport_error") from None
        finally:
            event["event"] = "request_result"
            event["elapsed_seconds"] = time.monotonic() - start
            try:
                self._emit(event)
            except ProviderError:
                if event["status"] == "completed":
                    # A failed result sink cannot be presented as a successful call.
                    self._events[-1]["status"] = "event_sink_error"
                    raise

    def _read_metadata(self, raw: dict[str, Any], event: dict[str, Any]) -> None:
        usage = raw.get("usage")
        if isinstance(usage, dict):
            event["usage"] = {
                key: usage[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if type(usage.get(key)) is int and usage[key] >= 0
            } or None
            event["usage_known"] = event["usage"] is not None and "total_tokens" in event["usage"]
        model = raw.get("model")
        if (
            isinstance(model, str)
            and _MODEL_NAME.fullmatch(model)
            and not (self._api_key and self._api_key in model)
        ):
            # Provider metadata is untrusted: never log an arbitrary returned string.
            event["response_model_sha256"] = hashlib.sha256(model.encode("utf-8")).hexdigest()
            event["response_model_matches_configured"] = model == self._model
            event["response_model"] = self._model if model == self._model else None
