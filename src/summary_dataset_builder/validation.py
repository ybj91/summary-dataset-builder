import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherResult:
    accepted: bool
    summary: str
    rejection_reason: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate response field")
        result[key] = value
    return result


def validate_teacher(content: str | None, finish_reason: str) -> TeacherResult:
    if finish_reason != "stop":
        raise ValueError("incomplete_completion:" + str(finish_reason))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty_completion")
    try:
        value = json.loads(content, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("invalid_json") from None
    if not isinstance(value, dict) or set(value) != {
        "accepted", "summary", "rejection_reason"
    }:
        raise ValueError("invalid_response_fields")
    if type(value["accepted"]) is not bool or not all(
        isinstance(value[key], str) for key in ("summary", "rejection_reason")
    ):
        raise ValueError("invalid_response_types")
    summary = value["summary"].strip()
    reason = value["rejection_reason"].strip()
    if value["accepted"]:
        if not summary or value["rejection_reason"] != "":
            raise ValueError("inconsistent_acceptance")
    elif value["summary"] != "" or not reason:
        raise ValueError("inconsistent_rejection")
    return TeacherResult(value["accepted"], summary, reason)


def screening_flags(source: str, summary: str) -> list[str]:
    """Conservative review signals, not a factuality score."""
    flags = []
    words = re.findall(r"\b[\w'-]+\b", summary)
    if len(words) > 110:
        flags.append("long_summary")
    if "\n\n" in summary or "\r\n\r\n" in summary:
        flags.append("multiple_paragraphs")
    numbers = lambda text: {
        value.replace(",", "")
        for value in re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|(?=\b))", text)
    }
    if numbers(summary) - numbers(source):
        flags.append("unsupported_number")
    sentences = [
        part.strip().casefold() for part in re.split(r"[.!?]+", summary) if part.strip()
    ]
    if len(sentences) != len(set(sentences)):
        flags.append("repeated_sentence")
    if re.search(
        r"\b(ignore (?:all |prior |previous )?instructions|system prompt|"
        r"ignore prior directions|as an ai|scanned (?:document|note))\b",
        summary, re.IGNORECASE
    ):
        flags.append("instruction_or_meta_content")
    return flags
