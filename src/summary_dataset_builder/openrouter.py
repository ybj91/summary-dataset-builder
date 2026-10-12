"""Single-attempt OpenRouter transport; callers must reserve budget before use.

No retries, generation lookups, credential files, or response-body logging occur
here. Unknown billing must be reconciled by the caller, not assumed to be free.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from http.client import HTTPException
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_TOKENS = 1_000_000
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_TOKEN_DETAILS = {
    "prompt_tokens_details": (
        "cached_tokens", "cache_write_tokens", "audio_tokens", "video_tokens",
    ),
    "completion_tokens_details": (
        "reasoning_tokens", "accepted_prediction_tokens",
        "rejected_prediction_tokens", "audio_tokens", "image_tokens",
    ),
}
_COST_DETAILS = (
    "upstream_inference_cost", "upstream_inference_prompt_cost",
    "upstream_inference_completions_cost",
)


@dataclass
class Completion:
    content: str | None
    finish_reason: str
    request_id: str
    model: str
    provider: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: Decimal | None
    usage: dict


class OpenRouterError(Exception):
    """Safe diagnostics only; uncertain means this attempt may have been billed."""

    def __init__(
        self,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        uncertain: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__("OpenRouter request failed.")
        self.status_code = status_code
        self.retryable = retryable
        self.uncertain = uncertain
        self.retry_after = retry_after


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise OpenRouterError() from None


def _validate_messages_schema(messages: list[dict[str, str]], schema: dict) -> None:
    if not isinstance(messages, list) or not messages:
        raise OpenRouterError()
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("system", "developer", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in message.items())
        ):
            raise OpenRouterError()
    if not isinstance(schema, dict) or not schema:
        raise OpenRouterError()


def estimate_input_tokens(messages: list[dict[str, str]], schema: dict) -> int:
    """Reserve one token per serialized UTF-8 byte, plus 4096 for framing.

    This is deliberately conservative, NOT a tokenizer count or a guaranteed
    bound on provider-added tokens. The caller still needs a spending limit.
    """
    _validate_messages_schema(messages, schema)
    try:
        return len(_json_text({"messages": messages, "schema": schema}).encode("utf-8")) + 4096
    except UnicodeError:
        raise OpenRouterError() from None


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 512
        and all(33 <= ord(character) <= 126 for character in value)
    )


def _nonnegative_cost(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        return None
    if isinstance(value, str) and not re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", value
    ):
        return None
    try:
        cost = Decimal(value)
    except (InvalidOperation, ValueError, OverflowError):
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _sanitize_usage(raw: dict) -> dict:
    """Keep only accounting fields; monetary values remain exact JSON strings."""
    usage = {}
    for field in _TOKEN_FIELDS:
        if field in raw:
            usage[field] = _token_count(raw[field])
    if "cost" in raw:
        cost = _nonnegative_cost(raw["cost"])
        usage["cost"] = str(cost) if cost is not None else None
    if type(raw.get("is_byok")) is bool:
        usage["is_byok"] = raw["is_byok"]
    for field, keys in _TOKEN_DETAILS.items():
        if isinstance(raw.get(field), dict):
            usage[field] = {
                key: _token_count(raw[field][key]) for key in keys if key in raw[field]
            }
    if isinstance(raw.get("cost_details"), dict):
        details = {}
        for key in _COST_DETAILS:
            if key in raw["cost_details"]:
                cost = _nonnegative_cost(raw["cost_details"][key])
                details[key] = str(cost) if cost is not None else None
        usage["cost_details"] = details
    return usage


def _retry_after(value: object) -> float | None:
    if not isinstance(value, str) or len(value) > 128:
        return None
    value = value.strip()
    try:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
            seconds = float(value)
        else:
            retry_date = parsedate_to_datetime(value)
            if retry_date.tzinfo is None:
                return None
            seconds = (retry_date - datetime.now(timezone.utc)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return None
    return min(60.0, max(0.0, seconds)) if math.isfinite(seconds) else None


def _status_error(status: int, headers) -> OpenRouterError:
    if status == 429:
        return OpenRouterError(
            status_code=status,
            retryable=True,
            uncertain=False,
            retry_after=_retry_after(headers.get("Retry-After")) if headers else None,
        )
    return OpenRouterError(
        status_code=status,
        uncertain=not (400 <= status < 500 and status != 408),
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member.")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError("Non-standard JSON constant.")


class OpenRouterClient:
    """Read OPENROUTER_API_KEY once; make at most one HTTP attempt per complete."""

    def __init__(self, *, timeout: float = 120):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key or any(not 33 <= ord(character) <= 126 for character in key):
            raise OpenRouterError()
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise OpenRouterError()
        try:
            valid_timeout = math.isfinite(timeout)
        except OverflowError:
            raise OpenRouterError() from None
        if not valid_timeout:
            raise OpenRouterError()
        self._key = key
        self._timeout = timeout
        self._opener = build_opener(_NoRedirect())

    def complete(
        self,
        *,
        model: str,
        providers: list[str],
        messages: list[dict[str, str]],
        schema: dict,
        max_tokens: int,
        input_price_per_million: Decimal,
        output_price_per_million: Decimal,
        reasoning_effort: str = "low",
    ) -> Completion:
        """Submit once with fixed routing and per-million-token price ceilings.

        max_tokens is an integer in [1, 1_000_000], including reasoning on most
        providers. A 2048-token budget with low effort is a practical starting
        point; empty/length completions are returned for caller classification.
        Model identifiers must match literally on return (aliases are rejected).
        Provider display names are preserved, not inferred from requested slugs.
        """
        if (
            not _identifier(model)
            or not isinstance(providers, list)
            or not providers
            or any(not _identifier(provider) for provider in providers)
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= _MAX_TOKENS
            or not isinstance(reasoning_effort, str)
            or reasoning_effort not in _EFFORTS
        ):
            raise OpenRouterError()
        for price in (input_price_per_million, output_price_per_million):
            if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
                raise OpenRouterError()
        _validate_messages_schema(messages, schema)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "teacher_response", "strict": True, "schema": schema},
            },
            "reasoning": {"effort": reasoning_effort},
            "temperature": 0,
            "stream": False,
        }
        routing = {
            "only": providers,
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
        # Validated Decimal literals avoid rounding a spending ceiling upward.
        price_json = (
            f'{{"prompt":{input_price_per_million},'
            f'"completion":{output_price_per_million}}}'
        )
        text = (
            _json_text(payload)[:-1] + ',"provider":'
            + _json_text(routing)[:-1] + ',"max_price":' + price_json + "}}"
        )
        try:
            body = text.encode("utf-8")
        except UnicodeError:
            raise OpenRouterError() from None
        request = Request(
            _ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            method="POST",
        )
        status = None
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                status = response.status
                if status != 200:
                    raise _status_error(status, response.headers)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                length = response.headers.get("Content-Length")
                if (
                    not isinstance(raw, bytes)
                    or len(raw) > _MAX_RESPONSE_BYTES
                    or (length is not None and (
                        not isinstance(length, str)
                        or len(length) > 20
                        or not re.fullmatch(r"[0-9]+", length)
                        or int(length) != len(raw)
                    ))
                ):
                    raise OpenRouterError(status_code=status, uncertain=True)
        except HTTPError as error:
            failure = _status_error(error.code, error.headers)
            try:
                error.close()
            except (OSError, HTTPException):
                pass
            raise failure from None
        except (URLError, OSError, HTTPException, OverflowError):
            raise OpenRouterError(status_code=status, uncertain=True) from None
        return self._parse_completion(raw, model)

    def _parse_completion(self, raw: bytes, model: str) -> Completion:
        try:
            envelope = json.loads(
                raw.decode("utf-8"),
                parse_float=Decimal,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (UnicodeError, ValueError, InvalidOperation, OverflowError, RecursionError):
            raise OpenRouterError(status_code=200, uncertain=True) from None
        if not isinstance(envelope, dict) or "error" in envelope:
            raise OpenRouterError(status_code=200, uncertain=True)
        choices = envelope.get("choices")
        if (
            not _identifier(envelope.get("id"))
            or envelope.get("model") != model
            or not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise OpenRouterError(status_code=200, uncertain=True)
        choice = choices[0]
        message = choice.get("message")
        if (
            "error" in choice
            or not _identifier(choice.get("finish_reason"))
            or ("index" in choice and (
                type(choice["index"]) is not int or choice["index"] != 0
            ))
            or not isinstance(message, dict)
            or "content" not in message
            or (message["content"] is not None and not isinstance(message["content"], str))
            or ("role" in message and message["role"] != "assistant")
        ):
            raise OpenRouterError(status_code=200, uncertain=True)
        raw_usage = envelope.get("usage")
        if raw_usage is not None and not isinstance(raw_usage, dict):
            raise OpenRouterError(status_code=200, uncertain=True)
        usage = _sanitize_usage(raw_usage or {})
        provider = envelope.get("provider")
        if provider is not None and not isinstance(provider, str):
            raise OpenRouterError(status_code=200, uncertain=True)
        provider = provider if provider and provider.strip() else None
        try:
            for value in (message["content"], provider):
                if value is not None:
                    value.encode("utf-8")
        except UnicodeError:
            raise OpenRouterError(status_code=200, uncertain=True) from None

        def redact(value: str | None) -> str | None:
            return value.replace(self._key, "[REDACTED]") if value is not None else None

        return Completion(
            content=redact(message["content"]),
            finish_reason=redact(choice["finish_reason"]),
            request_id=redact(envelope["id"]),
            model=redact(envelope["model"]),
            provider=redact(provider),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=_nonnegative_cost(usage.get("cost")),
            usage=usage,
        )
