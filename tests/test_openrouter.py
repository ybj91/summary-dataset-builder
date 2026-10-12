"""Offline transport tests. Every network entry point is mocked."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.message import Message
from email.utils import format_datetime
from http.client import IncompleteRead
import io
import json
import os
import socket
import traceback
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from urllib.response import addinfourl

from summary_dataset_builder.openrouter import (
    Completion,
    OpenRouterClient,
    OpenRouterError,
    estimate_input_tokens,
)


KEY = "test-only-secret-never-a-real-api-key"
MODEL = "example/teacher"


def completion_envelope():
    return {
        "id": "gen-offline-123",
        "model": MODEL,
        "provider": "Example Provider",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"summary":"A summary."}'},
        }],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 400,
            "total_tokens": 520,
            "cost": "0.0001234567890123456789",
            "completion_tokens_details": {"reasoning_tokens": 250},
        },
    }


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network_guard = patch(
            "socket.create_connection", side_effect=AssertionError("Network forbidden.")
        )
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)
        self.builder = patch("summary_dataset_builder.openrouter.build_opener")
        self.build_opener = self.builder.start()
        self.addCleanup(self.builder.stop)
        self.opener = self.build_opener.return_value
        self.client = OpenRouterClient(timeout=12.5)
        self.arguments = {
            "model": MODEL,
            "providers": ["example"],
            "messages": [{"role": "user", "content": "Summarize this document."}],
            "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            "max_tokens": 2048,
            "input_price_per_million": Decimal("0.123456789012345678901"),
            "output_price_per_million": Decimal("0.60"),
        }
        self.respond(completion_envelope())

    def respond(self, envelope=None, *, body=None, status=200, headers=None):
        if body is None:
            body = json.dumps(envelope).encode("utf-8")
        self.response = Mock()
        self.response.status = status
        self.response.headers = headers or {}
        self.response.read.return_value = body
        self.opener.open.side_effect = None
        self.opener.open.return_value.__enter__.return_value = self.response

    def complete(self, **overrides):
        return self.client.complete(**(self.arguments | overrides))

    def assert_safe_failure(self, operation, *, status=None, uncertain=False, retryable=False):
        with self.assertRaises(OpenRouterError) as caught:
            operation()
        error = caught.exception
        self.assertEqual(error.status_code, status)
        self.assertEqual(error.uncertain, uncertain)
        self.assertEqual(error.retryable, retryable)
        self.assertEqual(str(error), "OpenRouter request failed.")
        self.assertNotIn(KEY, repr(error))
        self.assertNotIn(KEY, "".join(traceback.format_exception(error)))
        return error

    def fail_http(self, status, retry_after=None):
        headers = Message()
        headers["X-Provider-Secret"] = KEY
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        body = io.BytesIO(f"Provider returned sensitive data {KEY}".encode())
        error = HTTPError("https://example.invalid/" + KEY, status, KEY, headers, body)
        self.opener.open.side_effect = error
        return body

    def test_request_has_fixed_headers_routing_schema_and_exact_price_caps(self):
        self.complete()
        self.opener.open.assert_called_once()
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + KEY)
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(request.get_header("Accept-encoding"), "identity")
        self.assertEqual(self.opener.open.call_args.kwargs, {"timeout": 12.5})
        payload = json.loads(request.data, parse_float=Decimal)
        self.assertNotIn(KEY, request.data.decode())
        self.assertEqual(payload["model"], MODEL)
        self.assertNotIn("models", payload)
        self.assertEqual(payload["provider"], {
            "only": ["example"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "max_price": {
                "prompt": self.arguments["input_price_per_million"],
                "completion": Decimal("0.60"),
            },
        })
        self.assertEqual(payload["messages"], self.arguments["messages"])
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["response_format"], {
            "type": "json_schema",
            "json_schema": {
                "name": "teacher_response",
                "strict": True,
                "schema": self.arguments["schema"],
            },
        })

    def test_success_preserves_precise_accounting(self):
        result = self.complete()
        self.assertIsInstance(result, Completion)
        self.assertEqual(result.content, '{"summary":"A summary."}')
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.request_id, "gen-offline-123")
        self.assertEqual(result.model, MODEL)
        self.assertEqual(result.provider, "Example Provider")
        self.assertEqual(result.prompt_tokens, 120)
        self.assertEqual(result.completion_tokens, 400)
        self.assertEqual(result.cost_usd, Decimal("0.0001234567890123456789"))
        self.assertEqual(result.usage["completion_tokens_details"]["reasoning_tokens"], 250)
        json.dumps(result.usage, allow_nan=False)

    def test_numeric_cost_is_parsed_without_float_rounding(self):
        body = json.dumps(completion_envelope()).replace(
            '"0.0001234567890123456789"', "0.0001234567890123456789"
        )
        self.respond(body=body.encode())
        self.assertEqual(self.complete().cost_usd, Decimal("0.0001234567890123456789"))

    def test_cost_unknown_is_not_assumed_zero(self):
        for value in (None, -1, "-0.01", "NaN", "Infinity", "abc", "", True, {}, [],
                      "1_000", " 1 "):
            with self.subTest(value=value):
                envelope = completion_envelope()
                envelope["usage"]["cost"] = value
                self.respond(envelope)
                result = self.complete()
                self.assertIsNone(result.cost_usd)
                self.assertIsNone(result.usage["cost"])
        envelope = completion_envelope()
        del envelope["usage"]["cost"]
        self.respond(envelope)
        self.assertIsNone(self.complete().cost_usd)

    def test_explicit_zero_cost_is_preserved(self):
        for value in (0, "0", "0.00000"):
            with self.subTest(value=value):
                envelope = completion_envelope()
                envelope["usage"]["cost"] = value
                self.respond(envelope)
                self.assertEqual(self.complete().cost_usd, Decimal(0))

    def test_missing_or_null_usage_is_unknown(self):
        for omit in (True, False):
            envelope = completion_envelope()
            if omit:
                del envelope["usage"]
            else:
                envelope["usage"] = None
            self.respond(envelope)
            result = self.complete()
            self.assertEqual(result.usage, {})
            self.assertIsNone(result.cost_usd)
            self.assertIsNone(result.prompt_tokens)
            self.assertIsNone(result.completion_tokens)

    def test_invalid_tokens_are_unknown_not_coerced(self):
        for value in (-1, 1.5, 12.0, True, "12", "NaN", None, [], {}):
            with self.subTest(value=value):
                envelope = completion_envelope()
                envelope["usage"]["prompt_tokens"] = value
                envelope["usage"]["completion_tokens"] = value
                envelope["usage"]["completion_tokens_details"]["reasoning_tokens"] = value
                self.respond(envelope)
                result = self.complete()
                self.assertIsNone(result.prompt_tokens)
                self.assertIsNone(result.completion_tokens)
                self.assertIsNone(result.usage["completion_tokens_details"]["reasoning_tokens"])

    def test_zero_tokens_are_valid(self):
        envelope = completion_envelope()
        envelope["usage"]["prompt_tokens"] = 0
        envelope["usage"]["completion_tokens"] = 0
        self.respond(envelope)
        result = self.complete()
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)

    def test_usage_retains_only_sanitized_accounting_details(self):
        envelope = completion_envelope()
        envelope["usage"].update({
            "raw_provider_response": KEY,
            "is_byok": True,
            "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": -1, KEY: KEY},
            "cost_details": {
                "upstream_inference_cost": "0.1",
                "upstream_inference_prompt_cost": "garbage " + KEY,
                "upstream_inference_completions_cost": "0.02",
                "raw_response": KEY,
            },
        })
        self.respond(envelope)
        usage = self.complete().usage
        self.assertNotIn(KEY, json.dumps(usage))
        self.assertNotIn("raw_provider_response", usage)
        self.assertEqual(usage["prompt_tokens_details"], {
            "cached_tokens": 10, "cache_write_tokens": None,
        })
        self.assertEqual(usage["cost_details"], {
            "upstream_inference_cost": "0.1",
            "upstream_inference_prompt_cost": None,
            "upstream_inference_completions_cost": "0.02",
        })
        self.assertTrue(usage["is_byok"])

    def test_successful_secret_echoes_are_redacted(self):
        envelope = completion_envelope()
        envelope["id"] = "gen-" + KEY
        envelope["provider"] = "provider " + KEY
        envelope["choices"][0]["finish_reason"] = KEY
        envelope["choices"][0]["message"]["content"] = "malicious echo " + KEY
        envelope["choices"][0]["message"]["reasoning"] = KEY
        self.respond(envelope)
        result = self.complete()
        self.assertNotIn(KEY, repr(result))
        self.assertEqual(result.content, "malicious echo [REDACTED]")

    def test_unknown_provider_is_not_inferred_from_request(self):
        for value in (None, "", " ", "Unrecognized Provider"):
            with self.subTest(value=value):
                envelope = completion_envelope()
                envelope["provider"] = value
                self.respond(envelope)
                self.assertEqual(self.complete().provider, value if value and value.strip() else None)
        envelope = completion_envelope()
        del envelope["provider"]
        self.respond(envelope)
        self.assertIsNone(self.complete().provider)

    def test_empty_null_and_truncated_content_preserve_billing_and_reason(self):
        for content in (None, "", '{"summary": "interrupted'):
            with self.subTest(content=content):
                envelope = completion_envelope()
                envelope["choices"][0]["finish_reason"] = "length"
                envelope["choices"][0]["message"]["content"] = content
                self.respond(envelope)
                result = self.complete()
                self.assertEqual(result.content, content)
                self.assertEqual(result.finish_reason, "length")
                self.assertEqual(result.request_id, "gen-offline-123")
                self.assertEqual(result.completion_tokens, 400)
                self.assertEqual(result.cost_usd, Decimal("0.0001234567890123456789"))

    def test_malformed_json_is_uncertain_and_redacted(self):
        for body in (b"", b"{", b"\xff", ("not-json " + KEY).encode(),
                     b'{"id":"a","id":"b"}', b"[" * 2000):
            with self.subTest(body=body[:20]):
                self.respond(body=body)
                self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_nonstandard_json_constants_are_uncertain_not_valid_zero_cost(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            envelope = completion_envelope()
            envelope["usage"]["cost"] = value
            self.respond(envelope)
            self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_malformed_envelopes_are_uncertain(self):
        envelopes = [None, [], 123, "raw secret " + KEY]
        for field, values in {
            "id": [None, "", 123],
            "model": [None, "", "another/model", MODEL + ":alias"],
            "choices": [None, [], [{}], [1], [completion_envelope()["choices"][0]] * 2],
            "usage": [[], "raw " + KEY, 0],
            "provider": [{}, 123],
        }.items():
            for value in values:
                envelope = completion_envelope()
                envelope[field] = value
                envelopes.append(envelope)
        for field in ("id", "model", "choices"):
            envelope = completion_envelope()
            del envelope[field]
            envelopes.append(envelope)
        for envelope in envelopes:
            with self.subTest(envelope=envelope):
                self.respond(envelope)
                self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_malformed_choices_are_uncertain(self):
        for field, values in {
            "finish_reason": [None, "", [], 5],
            "index": [1, True, "0"],
            "message": [None, {}, [], {"content": []}, {"content": 0},
                        {"content": "text", "role": "user"}],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    envelope = completion_envelope()
                    envelope["choices"][0][field] = value
                    self.respond(envelope)
                    self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_http_200_embedded_error_never_implies_free_retry(self):
        for value in (None, {"code": 429, "message": KEY}, {"code": 401, "message": KEY}):
            envelope = completion_envelope()
            envelope["error"] = value
            self.respond(envelope)
            error = self.assert_safe_failure(self.complete, status=200, uncertain=True)
            self.assertIsNone(error.retry_after)
        envelope = completion_envelope()
        envelope["choices"][0]["error"] = {"message": KEY}
        self.respond(envelope)
        self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_429_is_only_retryable_error_and_body_is_not_read(self):
        body = self.fail_http(429, "120")
        error = self.assert_safe_failure(
            self.complete, status=429, retryable=True, uncertain=False
        )
        self.assertEqual(error.retry_after, 60.0)
        self.assertTrue(body.closed)
        self.opener.open.assert_called_once()

    def test_retry_after_seconds_are_bounded_or_unknown(self):
        for value, expected in (
            ("0", 0.0), ("1.25", 1.25), ("59", 59.0), ("61", 60.0),
            ("-1", None), ("NaN", None), ("Infinity", None), ("nonsense", None),
            (KEY, None), (None, None), ("9" * 200, None),
        ):
            with self.subTest(value=value):
                self.fail_http(429, value)
                error = self.assert_safe_failure(self.complete, status=429, retryable=True)
                self.assertEqual(error.retry_after, expected)

    def test_retry_after_http_dates_are_bounded(self):
        now = datetime.now(timezone.utc)
        for date, expected in (
            (now + timedelta(hours=1), 60.0),
            (now - timedelta(hours=1), 0.0),
        ):
            self.fail_http(429, format_datetime(date, usegmt=True))
            error = self.assert_safe_failure(self.complete, status=429, retryable=True)
            self.assertEqual(error.retry_after, expected)

    def test_auth_and_other_4xx_are_known_fatal(self):
        for status in (400, 401, 402, 403, 404, 405, 413, 422):
            with self.subTest(status=status):
                self.fail_http(status, "60")
                error = self.assert_safe_failure(self.complete, status=status)
                self.assertIsNone(error.retry_after)

    def test_408_and_server_errors_are_uncertain_not_retryable(self):
        for status in (408, 500, 502, 503, 504, 529):
            with self.subTest(status=status):
                self.fail_http(status, "1")
                self.assert_safe_failure(self.complete, status=status, uncertain=True)

    def test_url_socket_and_timeout_errors_are_uncertain_without_retries(self):
        errors = (
            URLError(KEY), TimeoutError(KEY), socket.timeout(KEY),
            ConnectionResetError(KEY), OSError(KEY), IncompleteRead(KEY.encode(), 100),
        )
        for error in errors:
            with self.subTest(error_type=type(error).__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = error
                self.assert_safe_failure(self.complete, uncertain=True)
                self.opener.open.assert_called_once()

    def test_body_read_failure_is_uncertain_and_response_is_closed(self):
        for error in (TimeoutError(KEY), IncompleteRead(KEY.encode(), 100)):
            with self.subTest(error_type=type(error).__name__):
                self.response.read.side_effect = error
                self.assert_safe_failure(self.complete, status=200, uncertain=True)
                self.opener.open.return_value.__exit__.assert_called()

    def test_redirect_handler_never_creates_a_followup_request(self):
        handler = self.build_opener.call_args.args[0]
        self.assertIsInstance(handler, HTTPRedirectHandler)
        request = Request("https://openrouter.ai/api/v1/chat/completions", data=b"{}")
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                self.assertIsNone(handler.redirect_request(
                    request, None, status, KEY,
                    {"Location": "https://other.example/" + KEY},
                    "https://other.example/" + KEY,
                ))
                self.opener.open.reset_mock()
                self.fail_http(status)
                self.assert_safe_failure(self.complete, status=status, uncertain=True)
                self.opener.open.assert_called_once()

    def test_real_opener_rejects_redirect_before_sending_authorization_again(self):
        redirect_handler = self.build_opener.call_args.args[0]
        transport = HTTPSHandler()
        transport.https_open = Mock()
        self.client._opener = build_opener(redirect_handler, transport)
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                headers = Message()
                headers["Location"] = "https://other.example/" + KEY
                response = addinfourl(
                    io.BytesIO(KEY.encode()), headers,
                    "https://openrouter.ai/api/v1/chat/completions", status,
                )
                response.msg = "redirect " + KEY
                transport.https_open.reset_mock()
                transport.https_open.return_value = response
                self.assert_safe_failure(self.complete, status=status, uncertain=True)
                transport.https_open.assert_called_once()
                request = transport.https_open.call_args.args[0]
                self.assertEqual(request.host, "openrouter.ai")
                self.assertEqual(request.get_header("Authorization"), "Bearer " + KEY)

    def test_non_200_response_status_is_handled_without_reading_body(self):
        for status in (201, 204, 302, 401, 429, 503):
            with self.subTest(status=status):
                self.respond(body=KEY.encode(), status=status, headers={"Retry-After": "2"})
                self.assert_safe_failure(
                    self.complete, status=status, retryable=status == 429,
                    uncertain=status not in (401, 429),
                )
                self.response.read.assert_not_called()

    def test_body_size_is_bounded(self):
        self.respond(body=b"x" * (2 * 1024 * 1024 + 1))
        self.assert_safe_failure(self.complete, status=200, uncertain=True)
        self.response.read.assert_called_once_with(2 * 1024 * 1024 + 1)

    def test_content_length_detects_truncated_transfer_even_if_json_is_valid(self):
        body = json.dumps(completion_envelope()).encode()
        self.respond(body=body, headers={"Content-Length": str(len(body) + 20)})
        self.assert_safe_failure(self.complete, status=200, uncertain=True)
        self.respond(body=body, headers={"Content-Length": str(len(body))})
        self.assertEqual(self.complete().request_id, "gen-offline-123")

    def test_malformed_content_length_is_safe_and_uncertain(self):
        for length in (KEY, "-1", "9" * 5000, "10, 20"):
            self.respond(completion_envelope(), headers={"Content-Length": length})
            self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_escaped_invalid_unicode_is_an_uncertain_response(self):
        for field in ("content", "provider"):
            envelope = completion_envelope()
            if field == "content":
                envelope["choices"][0]["message"]["content"] = "\ud800"
            else:
                envelope["provider"] = "\ud800"
            self.respond(envelope)
            self.assert_safe_failure(self.complete, status=200, uncertain=True)

    def test_constructor_uses_only_environment_and_reads_key_once(self):
        with patch("builtins.open", side_effect=AssertionError("No credential files.")):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY}):
                client = OpenRouterClient()
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "changed-after-construction"}):
                client.complete(**self.arguments)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer " + KEY)
        self.assertNotIn(KEY, repr(client))

    def test_invalid_credentials_fail_before_network_and_do_not_echo(self):
        for key in ("", " ", KEY + " ", " " + KEY, KEY + "\tbad", KEY + "\nbad",
                    KEY + "\r", KEY + "\x00", KEY + "\x7f", KEY + "\u00a0",
                    KEY + "\u4e2d"):
            with self.subTest(key_kind=repr(key[-1:])):
                with patch("summary_dataset_builder.openrouter.os.environ.get", return_value=key):
                    self.assert_safe_failure(OpenRouterClient)
        with patch.dict(os.environ, {}, clear=True):
            self.assert_safe_failure(OpenRouterClient)
        self.opener.open.assert_not_called()

    def test_invalid_timeouts_fail_before_network(self):
        for timeout in (0, -1, 10**1000, float("nan"), float("inf"), True, None, "120"):
            with self.subTest(timeout=timeout):
                self.assert_safe_failure(lambda: OpenRouterClient(timeout=timeout))
        self.opener.open.assert_not_called()

    def test_invalid_request_parameters_fail_before_network(self):
        cases = {
            "model": ["", None, " ", "model\ninjection"],
            "providers": [[], None, "example", [""], [None], ["has space"]],
            "max_tokens": [0, -1, 1_000_001, True, 2.5, "2048"],
            "input_price_per_million": [Decimal(0), Decimal(-1), Decimal("NaN"),
                                        Decimal("sNaN"), Decimal("Infinity"), 1.0, "0.1"],
            "output_price_per_million": [Decimal(0), Decimal("-Infinity"), None],
            "reasoning_effort": ["invalid", "", None, []],
            "messages": [[], None, [{}], [{"role": "user", "content": None}],
                         [{"role": "invalid", "content": "x"}]],
            "schema": [None, [], {}, {"value": float("nan")}, {"value": object()}],
        }
        for parameter, values in cases.items():
            for value in values:
                with self.subTest(parameter=parameter, value=value):
                    self.assert_safe_failure(lambda: self.complete(**{parameter: value}))
        self.opener.open.assert_not_called()

    def test_invalid_unicode_and_recursive_schema_are_safe_preflight_errors(self):
        schema = {"type": "object"}
        schema["recursive"] = schema
        self.assert_safe_failure(lambda: self.complete(schema=schema))
        self.assert_safe_failure(lambda: self.complete(
            messages=[{"role": "user", "content": "\ud800"}]
        ))
        self.opener.open.assert_not_called()

    def test_supported_efforts_and_token_boundaries_are_passed_explicitly(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.complete(reasoning_effort=effort)
            payload = json.loads(self.opener.open.call_args.args[0].data)
            self.assertEqual(payload["reasoning"], {"effort": effort})
        for max_tokens in (1, 220, 2048, 1_000_000):
            self.complete(max_tokens=max_tokens)
            self.assertEqual(
                json.loads(self.opener.open.call_args.args[0].data)["max_tokens"], max_tokens
            )

    def test_input_estimate_uses_utf8_bytes_and_4096_margin(self):
        messages = [{"role": "user", "content": "Résumé \u4e2d\u6587 \U0001f642"}]
        schema = self.arguments["schema"]
        serialized = json.dumps(
            {"messages": messages, "schema": schema},
            ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(estimate_input_tokens(messages, schema), len(serialized) + 4096)
        bigger_schema = deepcopy(schema)
        bigger_schema["description"] = "More schema context"
        self.assertGreater(
            estimate_input_tokens(messages, bigger_schema),
            estimate_input_tokens(messages, schema),
        )
        self.opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
