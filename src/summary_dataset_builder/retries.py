"""Retry classification and shared reservation throttling."""


class RetryRateLimited(ValueError):
    def __init__(self, retry_after: float):
        super().__init__("Global retry interval has not elapsed")
        self.retry_after = retry_after


def retryable_error(error) -> bool:
    return (
        error.status_code == 429 and error.retryable
        or error.uncertain and (
            error.status_code in (None, 200, 408)
            or error.status_code is not None and error.status_code >= 500
        )
    )


def retryable_reason(reason: str) -> bool:
    if reason in ("transport_error", "interrupted_request", "http_200", "http_408", "http_429"):
        return True
    if reason.startswith("http_"):
        code = reason.removeprefix("http_")
        return code.isdigit() and 500 <= int(code) <= 599
    return reason.startswith("retry_exhausted:") or reason == "rate_limit_retry_exhausted"


def reserve_with_throttle(store, document_id, amount, *, kind, sleep):
    reserve = store.reserve if kind == "generation" else store.reserve_assessment
    while True:
        try:
            return reserve(document_id, amount)
        except RetryRateLimited as error:
            sleep(error.retry_after)
