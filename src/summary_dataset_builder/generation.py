import json
from decimal import Decimal
import time

from .openrouter import OpenRouterClient, OpenRouterError, estimate_input_tokens
from .retries import reserve_with_throttle, retryable_error
from .store import RunStore, money
from .validation import screening_flags, validate_teacher


def generate(
    store: RunStore, *, approve_paid: bool, max_requests: int = 100,
    client: OpenRouterClient | None = None, sleep=time.sleep,
) -> dict:
    if not approve_paid:
        raise ValueError("Paid generation requires --approve-paid on each invocation")
    if type(max_requests) is not int or not 1 <= max_requests <= 100000:
        raise ValueError("max_requests must be 1..100000")
    if store.status()["inflight_attempts"] or store.accounting()["blocking_uncertain_attempts"]:
        raise ValueError("Resolve in-flight/uncertain requests before generating")
    config = store.config
    request_count = 0
    stop_reason = "request_limit"
    for _ in range(max_requests):
        document = store.next_document()
        if document is None:
            stop_reason = (
                "targets_met" if store.status()["generation_targets_met"] else "candidates_exhausted"
            )
            break
        messages = [
            {"role": "system", "content": store.prompt},
            {"role": "user", "content": json.dumps({"document": document["text"]}, ensure_ascii=False)},
        ]
        input_price = money(config["input_price_per_million"])
        output_price = money(config["output_price_per_million"])
        reserve = (
            Decimal(estimate_input_tokens(messages, store.schema)) * input_price
            + Decimal(config["max_tokens"]) * output_price
        ) / Decimal(1000000)
        if client is None:
            client = OpenRouterClient()
        attempt = reserve_with_throttle(
            store, document["document_sha256"], reserve, kind="generation", sleep=sleep
        )
        request_count += 1
        try:
            completion = client.complete(
                model=config["model"], providers=config["providers"], messages=messages,
                schema=store.schema, max_tokens=config["max_tokens"],
                input_price_per_million=input_price, output_price_per_million=output_price,
                reasoning_effort=config["reasoning_effort"],
            )
        except OpenRouterError as error:
            if store.retry_policy is not None:
                reason = f"http_{error.status_code}" if error.status_code is not None else "transport_error"
                store.finish(
                    attempt, status="error", cost=None if error.uncertain else Decimal(0), reason=reason,
                )
                if retryable_error(error):
                    if error.retry_after is not None and request_count < max_requests:
                        sleep(min(60, max(0, error.retry_after)))
                    continue
                stop_reason = reason
                break
            tries = store.db.execute(
                "SELECT attempts FROM documents WHERE id=?", (document["document_sha256"],)
            ).fetchone()["attempts"]
            retry = error.retryable and not error.uncertain and tries < config["max_attempts"]
            reason = (
                "rate_limit_retry_exhausted" if error.retryable and not retry
                else f"http_{error.status_code}" if error.status_code is not None else "transport_error"
            )
            store.finish(
                attempt, status="pending" if retry else "error",
                cost=None if error.uncertain else Decimal(0), reason=reason,
            )
            if retry:
                sleep(min(60, error.retry_after if error.retry_after is not None else 2 ** tries))
                continue
            stop_reason = "uncertain_billing" if error.uncertain else reason
            break
        except KeyboardInterrupt:
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
            parsed = validate_teacher(completion.content, completion.finish_reason)
            flags = screening_flags(document["text"], parsed.summary) if parsed.accepted else []
            status = "flagged" if flags else "accepted" if parsed.accepted else "rejected"
            reason = ",".join(flags) if flags else parsed.rejection_reason
            result = {
                "accepted": parsed.accepted, "summary": parsed.summary,
                "rejection_reason": parsed.rejection_reason, "flags": flags,
            }
        except ValueError as error:
            status = "error"
            reason = str(error)
        store.finish(
            attempt, status=status, cost=completion.cost_usd, response=response, result=result, reason=reason,
        )
        if completion.cost_usd is None and store.retry_policy is None:
            stop_reason = "uncertain_billing"
            break
        if completion.cost_usd is not None and completion.cost_usd > reserve:
            stop_reason = "provider_cost_exceeded_reservation"
            break
    output = store.status()
    if output["generation_targets_met"] and stop_reason == "request_limit":
        stop_reason = "targets_met"
    output.update({"requests_this_invocation": request_count, "stop_reason": stop_reason})
    return output
