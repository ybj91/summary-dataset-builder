"""Paid, source-only screening kept separate from generation and human review.

Same-model critique does not establish independent verification or factual truth.
Local validation checks the response contract and quotes, not semantic accuracy.
"""

import json
from decimal import Decimal
from pathlib import Path
import time

from .openrouter import OpenRouterClient, OpenRouterError, estimate_input_tokens
from .retries import reserve_with_throttle, retryable_error
from .store import RunStore, money


_VERDICTS = ("clear", "concern", "uncertain")
_SUPPORT = ("supported", "unsupported", "uncertain", "not_applicable")
_COVERAGE = ("adequate", "missing_key_information", "not_applicable")
_CATEGORIES = (
    "unsupported_claim", "contradicted_claim", "missing_key_information",
    "attribution_or_uncertainty", "misleading_emphasis", "inappropriate_disposition",
    "technical_error", "screening_false_positive", "instruction_following",
    "uncertainty", "other",
)
_FIELDS = (
    "verdict", "source_support", "coverage", "disposition_correct", "issues", "explanation",
)
_UNSUPPORTED_CATEGORIES = {
    "unsupported_claim", "contradicted_claim", "attribution_or_uncertainty",
    "instruction_following",
}
_DISPOSITION_CATEGORIES = {"inappropriate_disposition", "screening_false_positive"}
_SUMMARY_CATEGORIES = _UNSUPPORTED_CATEGORIES | {
    "missing_key_information", "misleading_emphasis",
}
ASSESSMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_FIELDS),
    "properties": {
        "verdict": {"type": "string", "enum": list(_VERDICTS)},
        "source_support": {"type": "string", "enum": list(_SUPPORT)},
        "coverage": {"type": "string", "enum": list(_COVERAGE)},
        "disposition_correct": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "claim", "evidence", "explanation"],
                "properties": {
                    "category": {"type": "string", "enum": list(_CATEGORIES)},
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "explanation": {"type": "string"},
                },
            },
        },
        "explanation": {"type": "string"},
    },
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_field")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError("nonstandard_json_constant")


def _parse_json(content: str):
    return json.loads(
        content, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )


def _text(value: object, *, allow_empty: bool = False) -> bool:
    if not isinstance(value, str) or (not value.strip() and not (allow_empty and value == "")):
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def validate_assessment(
    content: str | None, finish_reason: str, *, source: str, disposition: str,
) -> dict:
    """Reject malformed or contradictory assessments without claiming factuality."""
    if disposition not in ("accepted", "flagged", "rejected", "error"):
        raise ValueError("invalid_original_disposition")
    if finish_reason != "stop":
        raise ValueError("incomplete_completion:" + str(finish_reason))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty_completion")
    try:
        value = _parse_json(content)
    except (ValueError, OverflowError, RecursionError):
        raise ValueError("invalid_json") from None
    if not isinstance(value, dict) or set(value) != set(_FIELDS):
        raise ValueError("invalid_assessment_fields")
    for key, choices in (
        ("verdict", _VERDICTS), ("source_support", _SUPPORT), ("coverage", _COVERAGE),
    ):
        if not isinstance(value[key], str) or value[key] not in choices:
            raise ValueError("invalid_assessment_" + key)
    if (
        type(value["disposition_correct"]) is not bool
        or not isinstance(value["issues"], list)
        or not _text(value["explanation"])
    ):
        raise ValueError("invalid_assessment_types")
    categories = set()
    for issue in value["issues"]:
        if not isinstance(issue, dict) or set(issue) != {
            "category", "claim", "evidence", "explanation",
        }:
            raise ValueError("invalid_issue_fields")
        if (
            not isinstance(issue["category"], str)
            or issue["category"] not in _CATEGORIES
            or not _text(issue["claim"])
            or not _text(issue["explanation"])
            or not _text(issue["evidence"], allow_empty=True)
        ):
            raise ValueError("invalid_issue_types")
        if issue["evidence"] and issue["evidence"] not in source:
            raise ValueError("evidence_not_in_source")
        categories.add(issue["category"])

    has_summary = disposition in ("accepted", "flagged")
    support = value["source_support"]
    coverage = value["coverage"]
    verdict = value["verdict"]
    if has_summary:
        if support == "not_applicable" or coverage == "not_applicable":
            raise ValueError("summary_assessment_not_applicable")
    elif (
        support != "not_applicable" or coverage != "not_applicable"
        or categories & _SUMMARY_CATEGORIES
    ):
        raise ValueError("assessment_invents_summary")
    if bool(categories & _UNSUPPORTED_CATEGORIES) != (support == "unsupported"):
        raise ValueError("inconsistent_source_support")
    if ("missing_key_information" in categories) != (coverage == "missing_key_information"):
        raise ValueError("inconsistent_coverage")
    if bool(categories & _DISPOSITION_CATEGORIES) != (not value["disposition_correct"]):
        raise ValueError("inconsistent_disposition")
    if "technical_error" in categories and disposition != "error":
        raise ValueError("technical_issue_without_error")
    if "screening_false_positive" in categories and (
        disposition != "flagged" or support != "supported"
    ):
        raise ValueError("inconsistent_screening_false_positive")
    if (
        disposition == "accepted" and value["disposition_correct"]
        and (support == "unsupported" or coverage == "missing_key_information")
    ):
        raise ValueError("incorrect_acceptance_marked_correct")
    if verdict == "clear":
        if (
            value["issues"] or not value["disposition_correct"]
            or (has_summary and (support != "supported" or coverage != "adequate"))
        ):
            raise ValueError("inconsistent_clear_verdict")
    elif not value["issues"]:
        raise ValueError("nonclear_verdict_requires_issues")
    if verdict == "uncertain" and "uncertainty" not in categories:
        raise ValueError("uncertain_verdict_requires_uncertainty")
    if support == "uncertain" and "uncertainty" not in categories:
        raise ValueError("uncertain_support_requires_uncertainty")
    if verdict == "concern" and categories == {"uncertainty"}:
        raise ValueError("uncertainty_without_established_concern")
    return value


def _assessment_input(document: dict) -> dict:
    result = document["result"] or {}
    value = {
        "source": document["record"]["text"],
        "original_disposition": document["status"],
        "original_decision": result.get("accepted"),
        "original_summary": result.get("summary"),
        "original_rejection_reason": result.get("rejection_reason"),
        "screening_flags": result.get("flags", []),
        "original_reason": document["reason"],
    }
    if document["status"] == "error":
        response = document["response"] or {}
        value["raw_completion"] = {
            "content": response.get("content"), "finish_reason": response.get("finish_reason"),
        }
    return value


def assess(
    store: RunStore, *, approve_paid: bool, max_requests: int = 100,
    prompt_path: Path, schema_path: Path,
    client: OpenRouterClient | None = None, sleep=time.sleep,
) -> dict:
    if not approve_paid:
        raise ValueError("Paid assessment requires --approve-paid on each invocation")
    if type(max_requests) is not int or not 1 <= max_requests <= 100000:
        raise ValueError("max_requests must be 1..100000")
    if store.status()["inflight_attempts"] or store.accounting()["blocking_uncertain_attempts"]:
        raise ValueError("Resolve in-flight/uncertain requests before assessing")
    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("Assessor prompt must not be empty")
    try:
        schema = _parse_json(schema_path.read_text(encoding="utf-8"))
    except (ValueError, OverflowError, RecursionError):
        raise ValueError("Invalid assessor schema JSON") from None
    if json.dumps(schema, sort_keys=True) != json.dumps(ASSESSMENT_SCHEMA, sort_keys=True):
        raise ValueError("Schema must describe the supported assessor response contract")
    store.freeze_assessment(prompt, schema, prompt_path.name)
    config = store.config
    input_price = money(config["input_price_per_million"])
    output_price = money(config["output_price_per_million"])
    request_count = 0
    stop_reason = "request_limit"
    for _ in range(max_requests):
        document = store.next_assessment()
        if document is None:
            stop_reason = "assessment_complete"
            break
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(_assessment_input(document), ensure_ascii=False)},
        ]
        reserve = (
            Decimal(estimate_input_tokens(messages, schema)) * input_price
            + Decimal(config["max_tokens"]) * output_price
        ) / Decimal(1000000)
        if client is None:
            client = OpenRouterClient()
        attempt = reserve_with_throttle(store, document["id"], reserve, kind="assessment", sleep=sleep)
        request_count += 1
        try:
            completion = client.complete(
                model=config["model"], providers=config["providers"], messages=messages,
                schema=schema, max_tokens=config["max_tokens"],
                input_price_per_million=input_price, output_price_per_million=output_price,
                reasoning_effort=config["reasoning_effort"],
            )
        except OpenRouterError as error:
            if store.retry_policy is not None:
                reason = f"http_{error.status_code}" if error.status_code is not None else "transport_error"
                store.finish_assessment(
                    attempt, status="error", cost=None if error.uncertain else Decimal(0), reason=reason,
                )
                if retryable_error(error):
                    if error.retry_after is not None and request_count < max_requests:
                        sleep(min(60, max(0, error.retry_after)))
                    continue
                stop_reason = reason
                break
            tries = store.assessment_attempt_count(document["id"])
            rate_limited = error.status_code == 429 and error.retryable and not error.uncertain
            retry = rate_limited and tries < min(3, config["max_attempts"])
            reason = (
                "rate_limit_retry_exhausted" if rate_limited and not retry
                else f"http_{error.status_code}" if error.status_code is not None else "transport_error"
            )
            store.finish_assessment(
                attempt, status="pending" if retry else "error",
                cost=None if error.uncertain else Decimal(0), reason=reason,
            )
            if retry:
                if request_count < max_requests:
                    sleep(min(60, max(0, error.retry_after if error.retry_after is not None else 2 ** tries)))
                continue
            stop_reason = "uncertain_billing" if error.uncertain else reason
            break
        except BaseException:
            store.recover(attempt)
            raise
        response = {
            "request_id": completion.request_id, "model": completion.model,
            "provider": completion.provider, "content": completion.content,
            "finish_reason": completion.finish_reason,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "cost_usd": str(completion.cost_usd) if completion.cost_usd is not None else None,
            "usage": completion.usage,
        }
        result = None
        reason = ""
        try:
            result = validate_assessment(
                completion.content, completion.finish_reason,
                source=document["record"]["text"], disposition=document["status"],
            )
            status = "completed"
        except ValueError as error:
            status = "error"
            reason = str(error)
        store.finish_assessment(
            attempt, status=status, cost=completion.cost_usd,
            response=response, result=result, reason=reason,
        )
        if completion.cost_usd is None and store.retry_policy is None:
            stop_reason = "uncertain_billing"
            break
        if completion.cost_usd is not None and completion.cost_usd > reserve:
            stop_reason = "provider_cost_exceeded_reservation"
            break
    output = store.assessment_status()
    if stop_reason == "request_limit" and store.next_assessment() is None:
        stop_reason = "assessment_complete"
    output.update({
        "requests_this_invocation": request_count, "stop_reason": stop_reason,
        "independent_verification_established": False, "human_review_performed": False,
    })
    return output
