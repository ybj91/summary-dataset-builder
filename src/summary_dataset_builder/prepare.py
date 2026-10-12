"""Bounded, offline source selection with disk-backed duplicate screening.

The scan considers only the first ``max_records`` physical JSONL records. Seeded
priorities are ranked on disk, with domain round-robin selection and three
length strata inside each domain. This is diversity-constrained sampling, not a
uniform sample of the entire source dataset.

Near duplicates use exact Jaccard similarity of normalized five-word shingles.
A SQLite inverted index of sorted shingle prefixes filters candidates without
an all-pairs comparison. Shingles have a fixed, corpus-independent lexical
order. Common prefixes can still produce many candidates on adversarial data.
Greedy representatives depend on input order; this is not transitive duplicate
clustering. Documents shorter than five words use one whole-document shingle.
Domain caps use lowercased hostnames with ``www.`` removed, not a public-suffix
list. Representatives later excluded by caps/sampling can still cause earlier
duplicate rejections; the algorithm favors leakage prevention over yield.

Screening catches some obvious boilerplate, email addresses, and credentials,
not all PII or unsafe content. Local ``sha256:`` revisions are checked against
the full input file. Hugging Face commit revisions are syntax-validated.
If an acquisition companion exists, its dataset/revision must match and its
file checksum is verified. Checksum verification reads all local snapshot
bytes, but JSON parsing remains bounded by ``max_records``. Local metadata is
not an independently authenticated claim about the upstream dataset.
Optional local-tokenizer screening measures the shared training prompt plus
a summary-token reserve before deduplication/splitting. The reserve is a
planning bound, not a guarantee that an eventual summary fits the context.
Preserved documents are authoritative anchors: their records and assignments
remain unchanged, they consume split/domain slots, and they enter the global
duplicate index before new sources. New-source quality/length screening is not
reapplied to anchors; schema, provenance, identity, duplicate, and optional
token-context violations in anchors fail the operation instead of dropping them.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
from typing import Any, BinaryIO, Mapping
import unicodedata
from urllib.parse import urlsplit
import uuid


_MAX_LINE_BYTES = 16 * 1024 * 1024
_SHA40 = re.compile(r"[0-9a-fA-F]{40}\Z")
_LOCAL_SHA = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")
_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|"
    r"password|authorization)[\"']?\s*[:=]\s*[\"']?"
    r"(?:Bearer\s+)?[a-z0-9_./+=-]{12,}"
    r"|\bsk-[a-z0-9_-]{16,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"\b(?:accept all cookies|manage cookie preferences|"
    r"enable javascript|subscribe to our newsletter|"
    r"all rights reserved|privacy policy|terms of service)\b"
)
_SPLITS = ("train", "validation", "test")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_revision(revision: str, *, allow_local: bool) -> None:
    if not isinstance(revision, str) or not (
        _SHA40.fullmatch(revision)
        or (allow_local and _LOCAL_SHA.fullmatch(revision))
    ):
        suffix = " or sha256: followed by 64 hexadecimal characters" if allow_local else ""
        raise ValueError(f"revision must be an immutable 40-character commit SHA{suffix}")


def _nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _reject_secret_path(path: Path) -> None:
    for candidate in (path, path.resolve()):
        name = candidate.name.casefold()
        compact = re.sub(r"[^a-z0-9]", "", candidate.stem.casefold())
        if (
            name == ".env"
            or name.startswith(".env.")
            or compact in {
                "apikey", "apikeys", "credentials", "credential",
                "secret", "secrets", "password", "passwords",
                "accesstoken", "token", "tokens",
            }
            or candidate.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"}
        ):
            raise ValueError("Refusing a secret or credential file path")


class _InvalidSourceUrl(ValueError):
    """A source URL failed validation, without exposing its contents."""


def _validate_source_record(record: Any, location: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{location}: record must be a JSON object")
    for field in ("text", "id"):
        if not isinstance(record.get(field), str):
            raise ValueError(f"{location}: required field '{field}' must be a string")
    if not record["id"].strip():
        raise ValueError(f"{location}: required field 'id' must not be empty")
    if not isinstance(record.get("url"), str):
        raise _InvalidSourceUrl(f"{location}: required field 'url' must be a string")
    try:
        url = urlsplit(record["url"])
        valid_url = (
            url.scheme.lower() in {"http", "https"}
            and bool(url.hostname)
            and url.username is None
            and url.password is None
            and not any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in record["url"]
            )
        )
        url.port
    except ValueError:
        valid_url = False
    if not valid_url:
        raise _InvalidSourceUrl(f"{location}: required field 'url' must be an HTTP(S) URL without credentials")
    return dict(record)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _read_acquisition_metadata(
    input_path: Path, dataset: str, revision: str,
) -> dict[str, Any] | None:
    metadata_path = input_path.with_name(input_path.name + ".metadata.json")
    if not _path_exists(metadata_path):
        return None
    _reject_secret_path(metadata_path)
    with metadata_path.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError(f"{metadata_path}: acquisition metadata exceeds the 64 KiB safety limit")
    try:
        metadata = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite_json)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValueError(f"{metadata_path}: invalid acquisition metadata JSON") from None
    if not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int or metadata["format_version"] != 1:
        raise ValueError(f"{metadata_path}: acquisition metadata requires format_version 1")
    for field in ("dataset", "revision", "subset", "split", "download_sha256"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"{metadata_path}: acquisition metadata requires string '{field}'")
    if not _SHA40.fullmatch(metadata["revision"]):
        raise ValueError(f"{metadata_path}: acquisition metadata revision is not a pinned commit SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", metadata["download_sha256"]):
        raise ValueError(f"{metadata_path}: acquisition metadata has an invalid download SHA256")
    if type(metadata.get("limit")) is not int or metadata["limit"] <= 0:
        raise ValueError(f"{metadata_path}: acquisition metadata limit must be positive")
    if (
        type(metadata.get("records_written")) is not int
        or not 0 <= metadata["records_written"] <= metadata["limit"]
    ):
        raise ValueError(f"{metadata_path}: acquisition metadata has an invalid record count")
    if metadata["dataset"] != dataset:
        raise ValueError(f"{metadata_path}: dataset does not match acquisition metadata")
    if metadata["revision"].casefold() != revision.casefold():
        raise ValueError(f"{metadata_path}: revision does not match acquisition metadata")
    validated = {
        key: metadata[key]
        for key in (
            "format_version", "dataset", "revision", "subset", "split",
            "limit", "records_written", "download_sha256",
        )
    }
    audit_fields = {
        "inspected_count", "downloaded_count", "rejected_count", "rejected_counts",
        "rejections_sha256", "rejections_file",
    }
    if audit_fields.intersection(metadata):
        if not audit_fields.issubset(metadata):
            raise ValueError(f"{metadata_path}: incomplete acquisition rejection metadata")
        for field in ("inspected_count", "downloaded_count", "rejected_count"):
            if type(metadata[field]) is not int or not 0 <= metadata[field] <= metadata["limit"]:
                raise ValueError(f"{metadata_path}: invalid acquisition metadata '{field}'")
        rejection_counts = metadata["rejected_counts"]
        if (
            not isinstance(rejection_counts, dict)
            or any(
                not isinstance(reason, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason)
                or type(count) is not int or count <= 0
                for reason, count in rejection_counts.items()
            )
            or sum(rejection_counts.values()) != metadata["rejected_count"]
            or metadata["downloaded_count"] != metadata["records_written"]
            or metadata["inspected_count"] != metadata["downloaded_count"] + metadata["rejected_count"]
        ):
            raise ValueError(f"{metadata_path}: inconsistent acquisition inspected/emitted/rejection counts")
        if (
            not isinstance(metadata["rejections_sha256"], str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", metadata["rejections_sha256"])
        ):
            raise ValueError(f"{metadata_path}: acquisition metadata has an invalid rejections SHA256")
        rejections_path = input_path.with_name(input_path.name + ".rejections.jsonl")
        if metadata["rejections_file"] != rejections_path.name:
            raise ValueError(f"{metadata_path}: acquisition rejection filename does not match")
        _reject_secret_path(rejections_path)
        if not rejections_path.is_file():
            raise ValueError(f"{metadata_path}: acquisition rejection sidecar is missing")
        rejection_hash = hashlib.sha256()
        rejection_lines = 0
        with rejections_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                rejection_hash.update(chunk)
                rejection_lines += chunk.count(b"\n")
        if rejection_hash.hexdigest() != metadata["rejections_sha256"].casefold():
            raise ValueError(f"{metadata_path}: rejection sidecar SHA256 does not match acquisition metadata")
        if rejection_lines != metadata["rejected_count"]:
            raise ValueError(f"{metadata_path}: rejection sidecar count does not match acquisition metadata")
        validated.update({field: metadata[field] for field in sorted(audit_fields)})
    return validated


def _write_json_line(handle: BinaryIO, value: Any) -> bytes:
    data = _json_bytes(value)
    handle.write(data)
    return data


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _shingles(words: list[str]) -> set[str]:
    if len(words) < 5:
        return {" ".join(words)}
    return {" ".join(words[index:index + 5]) for index in range(len(words) - 4)}


def _screen_record(
    record: dict[str, Any], normalized: str, words: list[str],
    min_words: int, max_words: int,
) -> str | None:
    if len(words) < min_words:
        return "length_too_short"
    if len(words) > max_words:
        return "length_too_long"
    screening_text = normalized + "\n" + json.dumps(record, ensure_ascii=False)
    if _EMAIL.search(screening_text):
        return "obvious_email"
    if _CREDENTIAL.search(screening_text):
        return "obvious_credential"
    boilerplate_words = sum(len(match.group().split()) for match in _BOILERPLATE.finditer(normalized))
    if boilerplate_words >= len(words) * 0.5:
        return "boilerplate"
    lines = [_normalize(line) for line in record["text"].splitlines() if line.strip()]
    if len(lines) >= 5:
        if Counter(lines).most_common(1)[0][1] >= len(lines) * 0.8:
            return "repeated_boilerplate"
    if len(words) >= 40 and len(set(words)) <= 4:
        return "repeated_boilerplate"
    return None


def _open_database(path: Path) -> sqlite3.Connection:
    database = sqlite3.connect(path)
    try:
        database.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -8192;
            CREATE TABLE fingerprints (
                normalized_sha256 TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                representative_source_id TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                normalized_sha256 TEXT NOT NULL,
                record TEXT NOT NULL,
                shingles TEXT NOT NULL,
                shingle_count INTEGER NOT NULL,
                domain TEXT NOT NULL,
                length_bucket INTEGER NOT NULL,
                sample_key REAL NOT NULL,
                split_key REAL NOT NULL,
                preserved_split TEXT,
                preserved_raw BLOB
            );
            CREATE INDEX documents_domain ON documents(domain);
            CREATE TABLE source_identities (
                source_id TEXT PRIMARY KEY,
                document_sha256 TEXT NOT NULL,
                source_url TEXT NOT NULL,
                preserved INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE prefixes (
                token TEXT NOT NULL,
                document_id INTEGER NOT NULL,
                PRIMARY KEY (token, document_id)
            ) WITHOUT ROWID;
            CREATE TABLE query_prefix (token TEXT PRIMARY KEY) WITHOUT ROWID;
            """
        )
    except BaseException:
        database.close()
        raise
    return database


def _find_near_duplicate(
    database: sqlite3.Connection, shingles: set[str], threshold: float,
) -> str | None:
    count = len(shingles)
    # A one-token safety margin avoids losing exact-boundary matches to rounding.
    prefix_length = min(count, count - math.ceil(threshold * count) + 2)
    prefix = sorted(shingles)[:prefix_length]
    database.execute("DELETE FROM query_prefix")
    database.executemany(
        "INSERT INTO query_prefix(token) VALUES (?)",
        ((token,) for token in prefix),
    )
    lower = max(1, math.ceil(threshold * count) - 1)
    sqlite_maximum = (1 << 63) - 1
    upper = (
        sqlite_maximum if threshold <= count / sqlite_maximum else
        min(sqlite_maximum, math.floor(count / threshold) + 1)
    )
    candidates = database.execute(
        """
        SELECT source_id, shingles FROM documents
        WHERE shingle_count BETWEEN ? AND ?
          AND id IN (
              SELECT p.document_id FROM query_prefix AS q
              CROSS JOIN prefixes AS p ON p.token = q.token
          )
        ORDER BY id
        """,
        (lower, upper),
    )
    for source_id, candidate_json in candidates:
        candidate = set(json.loads(candidate_json))
        intersection = len(shingles.intersection(candidate))
        union = count + len(candidate) - intersection
        if intersection / union >= threshold:
            return source_id
    return None


def _index_document(
    database: sqlite3.Connection, source_id: str, normalized_sha: str,
    record_json: str, shingles: set[str], domain: str, length_bucket: int,
    sample_key: float, split_key: float, threshold: float, *,
    preserved_split: str | None = None, preserved_raw: bytes | None = None,
) -> None:
    ordered_shingles = sorted(shingles)
    cursor = database.execute(
        "INSERT INTO documents(source_id, normalized_sha256, record, shingles, "
        "shingle_count, domain, length_bucket, sample_key, split_key, "
        "preserved_split, preserved_raw) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id, normalized_sha, record_json,
            json.dumps(ordered_shingles, ensure_ascii=False), len(shingles),
            domain, length_bucket, sample_key, split_key, preserved_split, preserved_raw,
        ),
    )
    prefix_length = min(len(shingles), len(shingles) - math.ceil(threshold * len(shingles)) + 2)
    database.executemany(
        "INSERT INTO prefixes(token, document_id) VALUES (?, ?)",
        ((token, cursor.lastrowid) for token in ordered_shingles[:prefix_length]),
    )


def _validate_preserved_record(
    record: Any, location: str, *, dataset: str, revision: str, license_name: str,
) -> tuple[dict[str, Any], str, list[str]]:
    if not isinstance(record, dict):
        raise ValueError(f"{location}: preserved record must be a JSON object")
    for field in (
        "source_id", "source_url", "source_dataset", "source_revision",
        "source_license", "document_sha256", "text", "split",
    ):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{location}: preserved field '{field}' must be a nonempty string")
    _validate_source_record(
        {"id": record["source_id"], "url": record["source_url"], "text": record["text"]},
        location,
    )
    for field, expected in (
        ("source_dataset", dataset), ("source_revision", revision), ("source_license", license_name),
    ):
        if record[field] != expected:
            raise ValueError(f"{location}: preserved {field} does not match preparation arguments")
    if record["split"] not in _SPLITS:
        raise ValueError(f"{location}: preserved split must be train, validation, or test")
    try:
        _json_bytes(record)
        document_sha = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        normalized = _normalize(record["text"])
        normalized_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError(f"{location}: preserved record cannot be encoded as UTF-8 JSON") from None
    if record["document_sha256"].casefold() != document_sha:
        raise ValueError(f"{location}: preserved document_sha256 does not match text")
    if "normalized_sha256" in record and (
        not isinstance(record["normalized_sha256"], str)
        or record["normalized_sha256"].casefold() != normalized_sha
    ):
        raise ValueError(f"{location}: preserved normalized_sha256 does not match text")
    return record, normalized_sha, normalized.split()


def _split_targets(selected: int, requested: dict[str, int]) -> dict[str, int]:
    total = sum(requested.values())
    if total == 0:
        return dict.fromkeys(_SPLITS, 0)
    targets = {split: selected * requested[split] // total for split in _SPLITS}
    remaining = selected - sum(targets.values())
    remainder_order = sorted(
        _SPLITS, key=lambda split: -(selected * requested[split] % total),
    )
    for split in remainder_order[:remaining]:
        targets[split] += 1
    return targets


def prepare_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    revision: str,
    license_name: str,
    seed: int = 42,
    train_count: int = 12000,
    validation_count: int = 300,
    test_count: int = 300,
    max_records: int = 100000,
    min_words: int = 120,
    max_words: int = 1000,
    near_duplicate_threshold: float = 0.8,
    max_per_domain: int = 200,
    tokenizer_path: Path | None = None,
    context_limit: int = 2048,
    summary_token_reserve: int = 256,
    preserve_path: Path | None = None,
) -> dict:
    """Prepare a new directory, returning the manifest plus output file paths.

    Every scanned record is either selected or has an ID-bearing rejection.
    Insufficient candidates produce proportional split counts, numeric
    ``shortfall``, and per-split shortfalls, not an exception. Invalid inputs
    fail with a line location and no source content. Lines larger than 16 MiB
    fail explicitly. SQLite bounds text/shingle storage to disk; query sorting
    can use memory proportional to candidate IDs, not the full source corpus.
    Acquisition metadata, when present, is mandatory provenance: mismatched
    dataset/revision or a changed snapshot causes failure, not an override.
    Local ``sha256:`` revisions must match the complete input file checksum.
    A supplied local tokenizer screens training-prompt tokens plus the summary
    reserve before any duplicate grouping or split assignment.
    ``preserve_path`` supplies complete prepared records, not raw source rows.
    Requested counts include preserved records. Anchors are emitted first in
    their original order, with raw JSON retained (a missing final newline is
    framed for JSONL). Remaining split quotas are filled proportionally when
    candidates are scarce. Source reencounters must match anchor text and URL;
    an incoming raw ``split`` remains upstream metadata, never an assignment.
    Scanned/eligible/rejected counters describe incoming sources. Selected and
    unique counts include anchors; explicit new-only counters are also returned.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    for name, value in (
        ("train_count", train_count), ("validation_count", validation_count),
        ("test_count", test_count), ("max_records", max_records),
        ("min_words", min_words), ("max_words", max_words),
        ("max_per_domain", max_per_domain),
        ("context_limit", context_limit), ("summary_token_reserve", summary_token_reserve),
    ):
        _positive_int(value, name)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if min_words > max_words:
        raise ValueError("min_words must not exceed max_words")
    if (
        isinstance(near_duplicate_threshold, bool)
        or not isinstance(near_duplicate_threshold, (int, float))
        or not math.isfinite(near_duplicate_threshold)
        or not 0 < near_duplicate_threshold <= 1
    ):
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    _validate_revision(revision, allow_local=True)
    _nonempty_string(dataset, "dataset")
    _nonempty_string(license_name, "license_name")
    _reject_secret_path(input_path)
    if _path_exists(output_dir):
        raise FileExistsError(f"Output already exists: {output_dir}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if preserve_path is not None:
        preserve_path = Path(preserve_path)
        _reject_secret_path(preserve_path)
        if not preserve_path.is_file():
            raise FileNotFoundError(f"Preserved documents JSONL does not exist: {preserve_path}")
    acquisition_metadata = _read_acquisition_metadata(input_path, dataset, revision)
    tokenizer = None
    tokenizer_sha256 = None
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)
        _reject_secret_path(tokenizer_path)
        if not tokenizer_path.is_file():
            raise ValueError("tokenizer_path must be a local tokenizer file; no downloads are performed")
        from .tokenization import LocalTokenizer, training_prompt

        tokenizer = LocalTokenizer(tokenizer_path)
        tokenizer_hash = hashlib.sha256()
        with tokenizer_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                tokenizer_hash.update(chunk)
        tokenizer_sha256 = tokenizer_hash.hexdigest()

    arguments = {
        "input_path": str(input_path), "output_dir": str(output_dir),
        "dataset": dataset, "revision": revision, "license_name": license_name,
        "seed": seed, "train_count": train_count,
        "validation_count": validation_count, "test_count": test_count,
        "max_records": max_records, "min_words": min_words,
        "max_words": max_words, "near_duplicate_threshold": near_duplicate_threshold,
        "max_per_domain": max_per_domain,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "tokenizer_sha256": tokenizer_sha256,
        "context_limit": context_limit, "summary_token_reserve": summary_token_reserve,
        "preserve_path": str(preserve_path) if preserve_path is not None else None,
    }
    requested = dict(zip(_SPLITS, (train_count, validation_count, test_count)))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    work_dir.mkdir()
    database: sqlite3.Connection | None = None
    rejected_counts: dict[str, int] = {}
    screening_counts: dict[str, int] = {}
    scanned = 0
    unique = 0
    eligible = 0
    preserved_split_counts = dict.fromkeys(_SPLITS, 0)
    preserved_count = 0
    preserved_documents_sha256 = None
    source_revision_verified = False
    sample_random = random.Random(seed)
    split_random = random.Random(seed ^ 0x5F3759DF)
    try:
        database = _open_database(work_dir / "selection.sqlite3")
        if preserve_path is not None:
            preserved_hash = hashlib.sha256()
            with preserve_path.open("rb") as anchors:
                for line_number in range(1, sum(requested.values()) + 2):
                    raw = anchors.readline(_MAX_LINE_BYTES + 1)
                    if not raw:
                        break
                    location = f"{preserve_path}:{line_number}"
                    if len(raw) > _MAX_LINE_BYTES:
                        raise ValueError(f"{location}: preserved JSONL record exceeds the 16 MiB safety limit")
                    preserved_hash.update(raw)
                    try:
                        record = json.loads(
                            raw.decode("utf-8"), parse_constant=_reject_nonfinite_json,
                            object_pairs_hook=_unique_json_object,
                        )
                    except (UnicodeDecodeError, ValueError, RecursionError):
                        raise ValueError(f"{location}: invalid preserved UTF-8 JSON record") from None
                    record, normalized_sha, words = _validate_preserved_record(
                        record, location, dataset=dataset, revision=revision, license_name=license_name,
                    )
                    if database.execute(
                        "SELECT 1 FROM source_identities WHERE source_id = ?", (record["source_id"],),
                    ).fetchone() is not None:
                        raise ValueError(f"{location}: duplicate preserved source_id")
                    split = record["split"]
                    if preserved_split_counts[split] >= requested[split]:
                        raise ValueError(f"{location}: preserved split count exceeds requested '{split}' count")
                    domain = urlsplit(record["source_url"]).hostname.casefold().removeprefix("www.")
                    domain_count = database.execute(
                        "SELECT COUNT(*) FROM documents WHERE domain = ?", (domain,),
                    ).fetchone()[0]
                    if domain_count >= max_per_domain:
                        raise ValueError(f"{location}: preserved domain count exceeds max_per_domain")
                    if tokenizer is not None:
                        try:
                            source_prompt_tokens = tokenizer.count(training_prompt(record["text"]))
                        except Exception:
                            raise ValueError(f"{location}: local tokenizer failed to count preserved training prompt") from None
                        if source_prompt_tokens + summary_token_reserve > context_limit:
                            raise ValueError(f"{location}: preserved source_context_overflow")
                    if database.execute(
                        "SELECT 1 FROM fingerprints WHERE normalized_sha256 = ?", (normalized_sha,),
                    ).fetchone() is not None:
                        raise ValueError(f"{location}: exact_duplicate among preserved documents")
                    shingles = _shingles(words)
                    if _find_near_duplicate(database, shingles, near_duplicate_threshold) is not None:
                        raise ValueError(f"{location}: near_duplicate among preserved documents")
                    database.execute(
                        "INSERT INTO fingerprints VALUES (?, ?, ?)",
                        (normalized_sha, record["source_id"], record["source_id"]),
                    )
                    database.execute(
                        "INSERT INTO source_identities VALUES (?, ?, ?, 1)",
                        (record["source_id"], record["document_sha256"].casefold(), record["source_url"]),
                    )
                    bucket = max(0, min(2, (len(words) - min_words) * 3 // (max_words - min_words + 1)))
                    _index_document(
                        database, record["source_id"], normalized_sha,
                        _json_bytes(record).decode("utf-8").rstrip("\n"), shingles,
                        domain, bucket, 0.0, 0.0, near_duplicate_threshold,
                        preserved_split=split, preserved_raw=raw,
                    )
                    preserved_split_counts[split] += 1
                    preserved_count += 1
            preserved_documents_sha256 = preserved_hash.hexdigest()
        remaining_requested = {
            split: requested[split] - preserved_split_counts[split] for split in _SPLITS
        }
        with input_path.open("rb") as source, (work_dir / "rejections.jsonl").open("wb") as rejected:
            if acquisition_metadata is not None or revision.startswith("sha256:"):
                snapshot_hash = hashlib.sha256()
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    snapshot_hash.update(chunk)
                snapshot_digest = snapshot_hash.hexdigest()
                if (
                    acquisition_metadata is not None
                    and snapshot_digest != acquisition_metadata["download_sha256"].casefold()
                ):
                    raise ValueError(f"{input_path}: file SHA256 does not match acquisition metadata")
                if revision.startswith("sha256:"):
                    if snapshot_digest != revision.removeprefix("sha256:").casefold():
                        raise ValueError(f"{input_path}: file SHA256 does not match local snapshot revision")
                    source_revision_verified = True
                source.seek(0)

            def reject(source_id: str, reason: str, **details: Any) -> None:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                _write_json_line(
                    rejected, {"source_id": source_id, "reason": reason, **details},
                )

            for line_number in range(1, max_records + 1):
                raw = source.readline(_MAX_LINE_BYTES + 1)
                if not raw:
                    break
                scanned += 1
                location = f"{input_path}:{line_number}"
                if len(raw) > _MAX_LINE_BYTES:
                    raise ValueError(f"{location}: JSONL record exceeds the 16 MiB safety limit")
                try:
                    record = json.loads(
                        raw.decode("utf-8"),
                        parse_constant=_reject_nonfinite_json,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
                    raise ValueError(f"{location}: invalid UTF-8 JSON record") from None
                record = _validate_source_record(record, location)
                if (
                    acquisition_metadata is not None
                    and "source_split" in record
                    and record["source_split"] != acquisition_metadata["split"]
                ):
                    raise ValueError(f"{location}: source_split does not match acquisition metadata")
                try:
                    record_json = _json_bytes(record).decode("utf-8").rstrip("\n")
                except (ValueError, UnicodeError, RecursionError):
                    raise ValueError(f"{location}: record cannot be encoded as UTF-8 JSON") from None
                document_sha = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                identity = database.execute(
                    "SELECT document_sha256, source_url, preserved FROM source_identities WHERE source_id = ?",
                    (record["id"],),
                ).fetchone()
                if identity is not None:
                    if identity[0] != document_sha or identity[1] != record["url"]:
                        raise ValueError(f"{location}: source_id reencounter has different content or source URL")
                    if identity[2]:
                        reject(record["id"], "existing_source")
                        continue
                else:
                    database.execute(
                        "INSERT INTO source_identities VALUES (?, ?, ?, 0)",
                        (record["id"], document_sha, record["url"]),
                    )
                normalized = _normalize(record["text"])
                words = normalized.split()
                screening_reason = _screen_record(record, normalized, words, min_words, max_words)
                if screening_reason:
                    screening_counts[screening_reason] = screening_counts.get(screening_reason, 0) + 1
                    reject(record["id"], "source_screening", screening_reason=screening_reason)
                    continue
                if tokenizer is not None:
                    try:
                        source_prompt_tokens = tokenizer.count(training_prompt(record["text"]))
                    except Exception:
                        raise ValueError(f"{location}: local tokenizer failed to count the training prompt") from None
                    if source_prompt_tokens + summary_token_reserve > context_limit:
                        reject(
                            record["id"], "source_context_overflow",
                            source_prompt_tokens=source_prompt_tokens,
                            summary_token_reserve=summary_token_reserve,
                            required_context_tokens=source_prompt_tokens + summary_token_reserve,
                            context_limit=context_limit,
                            tokenizer_sha256=tokenizer_sha256,
                        )
                        continue
                eligible += 1
                normalized_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                exact = database.execute(
                    "SELECT source_id, representative_source_id FROM fingerprints "
                    "WHERE normalized_sha256 = ?", (normalized_sha,),
                ).fetchone()
                if exact is not None:
                    reject(
                        record["id"], "exact_duplicate", duplicate_of=exact[0],
                        representative_source_id=exact[1],
                    )
                    continue
                shingles = _shingles(words)
                near = _find_near_duplicate(database, shingles, near_duplicate_threshold)
                database.execute(
                    "INSERT INTO fingerprints VALUES (?, ?, ?)",
                    (normalized_sha, record["id"], near if near is not None else record["id"]),
                )
                if near is not None:
                    reject(record["id"], "near_duplicate", duplicate_of=near)
                    continue
                domain = urlsplit(record["url"]).hostname.casefold().removeprefix("www.")
                bucket = min(2, (len(words) - min_words) * 3 // (max_words - min_words + 1))
                _index_document(
                    database, record["id"], normalized_sha, record_json, shingles,
                    domain, bucket, sample_random.random(), split_random.random(),
                    near_duplicate_threshold,
                )
                unique += 1
                if scanned % 1000 == 0:
                    database.commit()
            if acquisition_metadata is not None and scanned != min(
                max_records,
                acquisition_metadata.get("downloaded_count", acquisition_metadata["records_written"]),
            ):
                raise ValueError(f"{input_path}: scanned record count does not match acquisition metadata")
            database.commit()
            database.executescript(
                """
                CREATE TABLE ranked AS
                WITH preserved_domains AS (
                    SELECT domain, COUNT(*) AS count FROM documents
                    WHERE preserved_split IS NOT NULL GROUP BY domain
                ), preserved_lengths AS (
                    SELECT domain, length_bucket, COUNT(*) AS count FROM documents
                    WHERE preserved_split IS NOT NULL GROUP BY domain, length_bucket
                ), length_ranked AS (
                    SELECT d.id, d.domain, d.sample_key,
                        ROW_NUMBER() OVER (
                            PARTITION BY d.domain, d.length_bucket ORDER BY d.sample_key, d.id
                        ) + COALESCE(p.count, 0) AS length_rank
                    FROM documents AS d LEFT JOIN preserved_lengths AS p
                        ON p.domain = d.domain AND p.length_bucket = d.length_bucket
                    WHERE d.preserved_split IS NULL
                ), domain_ranked AS (
                    SELECT id, domain, ROW_NUMBER() OVER (
                        PARTITION BY domain ORDER BY length_rank, sample_key, id
                    ) AS domain_rank
                    FROM length_ranked
                )
                SELECT r.id, r.domain_rank + COALESCE(p.count, 0) AS domain_rank
                FROM domain_ranked AS r LEFT JOIN preserved_domains AS p ON p.domain = r.domain;
                CREATE UNIQUE INDEX ranked_id ON ranked(id);
                """
            )
            database.execute(
                "CREATE TABLE selected AS SELECT d.id FROM documents AS d "
                "JOIN ranked AS r ON r.id = d.id WHERE r.domain_rank <= ? "
                "ORDER BY r.domain_rank, d.sample_key, d.id LIMIT ?",
                (
                    min(max_per_domain, unique + preserved_count),
                    min(sum(remaining_requested.values()), unique),
                ),
            )
            database.execute("CREATE UNIQUE INDEX selected_id ON selected(id)")
            new_selected_count = database.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
            selected_count = preserved_count + new_selected_count
            new_split_counts = _split_targets(new_selected_count, remaining_requested)
            split_counts = {
                split: preserved_split_counts[split] + new_split_counts[split] for split in _SPLITS
            }
            for source_id, domain_rank in database.execute(
                "SELECT d.source_id, r.domain_rank FROM documents AS d "
                "JOIN ranked AS r ON r.id = d.id "
                "WHERE NOT EXISTS (SELECT 1 FROM selected AS s WHERE s.id = d.id) "
                "ORDER BY d.id"
            ):
                reject(
                    source_id,
                    "domain_cap" if domain_rank > max_per_domain else "excluded_sampling",
                )
            rejected.flush()
            os.fsync(rejected.fileno())

        document_hash = hashlib.sha256()
        train_end = new_split_counts["train"]
        validation_end = train_end + new_split_counts["validation"]
        with (work_dir / "documents.jsonl").open("wb") as documents:
            for (raw,) in database.execute(
                "SELECT preserved_raw FROM documents WHERE preserved_split IS NOT NULL ORDER BY id"
            ):
                framed = raw if raw.endswith(b"\n") else raw + b"\n"
                documents.write(framed)
                document_hash.update(framed)
            for index, (record_json, normalized_sha) in enumerate(database.execute(
                "SELECT d.record, d.normalized_sha256 FROM documents AS d "
                "JOIN selected AS s ON s.id = d.id ORDER BY d.split_key, d.id"
            )):
                record = json.loads(record_json)
                document = {
                    key: value for key, value in record.items()
                    if key not in {"id", "url", "text", "split"}
                }
                source_split = record.get("source_split", record.get("split"))
                if source_split is not None:
                    document["source_split"] = source_split
                document.update(
                    source_id=record["id"], source_url=record["url"],
                    source_dataset=dataset, source_revision=revision,
                    source_license=license_name,
                    document_sha256=hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
                    normalized_sha256=normalized_sha, text=record["text"],
                    split=(
                        "train" if index < train_end else
                        "validation" if index < validation_end else "test"
                    ),
                )
                document_hash.update(_write_json_line(documents, document))
            documents.flush()
            os.fsync(documents.fileno())
        database.close()
        database = None
        (work_dir / "selection.sqlite3").unlink()
        manifest = {
            "format_version": 1, "arguments": arguments,
            "scanned_records": scanned, "eligible_records": eligible,
            "unique_records": unique + preserved_count, "selected_records": selected_count,
            "new_unique_records": unique, "new_selected_records": new_selected_count,
            "preserved_documents_sha256": preserved_documents_sha256,
            "preserved_count": preserved_count, "preserved_split_counts": preserved_split_counts,
            "scan_limit_reached": scanned == max_records,
            "requested_split_counts": requested, "split_counts": split_counts,
            "shortfall": sum(requested.values()) - selected_count,
            "split_shortfalls": {
                split: requested[split] - split_counts[split] for split in _SPLITS
            },
            "documents_sha256": document_hash.hexdigest(),
            "rejected_counts": rejected_counts, "screening_counts": screening_counts,
            "selection_algorithm": "seeded_domain_round_robin_with_three_length_strata",
            "near_duplicate_algorithm": "greedy_exact_five_word_jaccard_with_prefix_index",
            "global_dedup_invariants": {
                "scope": "preserved_and_new", "anchors_indexed_first": True,
                "unique_source_ids": True, "unique_normalized_hashes": True,
                "pairwise_jaccard_below_threshold": True,
                "near_duplicate_threshold": near_duplicate_threshold,
            },
            "source_revision_verified": source_revision_verified,
            "acquisition_metadata": acquisition_metadata,
            "acquisition_metadata_verified": acquisition_metadata is not None,
            "screening_is_complete_pii_detection": False,
        }
        with (work_dir / "manifest.json").open("wb") as handle:
            _write_json_line(handle, manifest)
            handle.flush()
            os.fsync(handle.fileno())
        if _path_exists(output_dir):
            raise FileExistsError(f"Output already exists: {output_dir}")
        os.rename(work_dir, output_dir)
        return {
            **manifest, "output_dir": str(output_dir),
            "documents_path": str(output_dir / "documents.jsonl"),
            "manifest_path": str(output_dir / "manifest.json"),
            "rejections_path": str(output_dir / "rejections.jsonl"),
        }
    finally:
        if database is not None:
            database.close()
        if work_dir.exists():
            shutil.rmtree(work_dir)
