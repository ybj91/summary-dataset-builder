"""Explicit, revision-pinned acquisition; importing this module is offline.

``datasets`` is optional and imported only inside ``download_fineweb``.
The caller must obtain download approval before invoking the function.
``max_records`` limits inspected records, including URL rejections, not bytes:
the datasets implementation may download/cache a shard to yield those records.
Invalid URLs are excluded from the output and audited in the companion
``<output filename>.rejections.jsonl`` without their URL or document text.
The companion ``<output filename>.metadata.json`` is published last and acts
as the completion marker. Handled failures remove files published by this call;
process termination between publications can leave orphaned data or rejection
files, which must not be treated as a completed acquisition.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import os
from pathlib import Path
import re
from typing import Any
import uuid

from .prepare import (
    _CREDENTIAL,
    _InvalidSourceUrl,
    _nonempty_string,
    _path_exists,
    _positive_int,
    _reject_secret_path,
    _validate_revision,
    _validate_source_record,
    _write_json_line,
)


def _safe_rejection_id(source_id: str) -> tuple[str, bool]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", source_id) and not _CREDENTIAL.search(source_id):
        return source_id, False
    digest = hashlib.sha256(source_id.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"sha256:{digest}", True


def _publish_exclusive(staged: Path, destination: Path) -> None:
    # A same-filesystem hard link atomically publishes without replacing a racer.
    os.link(staged, destination)


def download_fineweb(
    output_path: Path,
    *,
    revision: str,
    max_records: int,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    split: str = "train",
) -> dict:
    """Inspect at most N records, publishing JSONL, URL audit, and metadata.

    Malformed/missing URLs are source rejections, not fatal stream errors.
    Missing text/IDs and unexpected row shapes still fail. Other JSON metadata
    is retained and ``source_split`` records the requested upstream split.
    Unsafe rejection IDs are replaced with their stable SHA256. The returned
    ``shortfall`` is the requested record limit minus emitted valid records.
    No resume or approval inference is performed. The output filesystem must
    support hard links for exclusive, atomic file publication.
    """
    output_path = Path(output_path)
    _validate_revision(revision, allow_local=False)
    _positive_int(max_records, "max_records")
    for name, value in (("dataset", dataset), ("subset", subset), ("split", split)):
        _nonempty_string(value, name)
    _reject_secret_path(output_path)
    metadata_path = output_path.with_name(output_path.name + ".metadata.json")
    rejections_path = output_path.with_name(output_path.name + ".rejections.jsonl")
    for path in (output_path, rejections_path, metadata_path):
        if _path_exists(path):
            raise FileExistsError(f"Output already exists: {path}")
    try:
        datasets = importlib.import_module("datasets")
    except ImportError:
        raise RuntimeError(
            "Acquisition requires the optional 'datasets' package; install it before downloading"
        ) from None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    data_stage = output_path.parent / f".{output_path.name}.building-{token}"
    metadata_stage = output_path.parent / f".{metadata_path.name}.building-{token}"
    rejections_stage = output_path.parent / f".{rejections_path.name}.building-{token}"
    published: list[Path] = []
    count = 0
    inspected_count = 0
    rejected_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    rejections_digest = hashlib.sha256()
    try:
        try:
            stream = datasets.load_dataset(
                dataset, name=subset, split=split, revision=revision, streaming=True,
            )
            rows = iter(itertools.islice(stream, max_records))
        except Exception as error:
            raise RuntimeError(
                f"Dataset stream initialization failed ({type(error).__name__}); no output published"
            ) from None
        with data_stage.open("xb") as output, rejections_stage.open("xb") as rejected:
            while True:
                try:
                    row = next(rows)
                except StopIteration:
                    break
                except Exception as error:
                    raise RuntimeError(
                        f"Dataset stream failed before record {inspected_count + 1} "
                        f"({type(error).__name__}); no output published"
                    ) from None
                inspected_count += 1
                location = f"dataset record {inspected_count}"
                try:
                    record = _validate_source_record(row, location)
                except _InvalidSourceUrl:
                    source_id, source_id_is_hash = _safe_rejection_id(row["id"])
                    reason = "invalid_source_url"
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                    rejections_digest.update(_write_json_line(rejected, {
                        "source_id": source_id, "source_id_is_hash": source_id_is_hash,
                        "ordinal": inspected_count, "reason": reason,
                    }))
                    continue
                record["source_split"] = split
                try:
                    digest.update(_write_json_line(output, record))
                except (TypeError, ValueError, UnicodeError, RecursionError):
                    raise ValueError(f"{location}: metadata must be UTF-8 JSON serializable") from None
                count += 1
            output.flush()
            os.fsync(output.fileno())
            rejected.flush()
            os.fsync(rejected.fileno())
        rejected_count = sum(rejected_counts.values())
        metadata: dict[str, Any] = {
            "format_version": 1, "dataset": dataset, "revision": revision,
            "subset": subset, "split": split, "limit": max_records,
            "records_written": count, "downloaded_count": count,
            "inspected_count": inspected_count, "rejected_count": rejected_count,
            "rejected_counts": rejected_counts,
            "record_limit_reached": inspected_count == max_records,
            "record_limit_kind": "inspected_records",
            "download_sha256": digest.hexdigest(),
            "rejections_sha256": rejections_digest.hexdigest(),
            "rejections_file": rejections_path.name,
            "shortfall": max_records - count,
            "all_records_rejected": inspected_count > 0 and count == 0,
            "record_limit_is_byte_limit": False,
        }
        with metadata_stage.open("xb") as output:
            _write_json_line(output, metadata)
            output.flush()
            os.fsync(output.fileno())
        _publish_exclusive(data_stage, output_path)
        published.append(output_path)
        _publish_exclusive(rejections_stage, rejections_path)
        published.append(rejections_path)
        _publish_exclusive(metadata_stage, metadata_path)
        published.append(metadata_path)
        return {
            **metadata, "output_path": str(output_path),
            "metadata_path": str(metadata_path), "rejections_path": str(rejections_path),
        }
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        data_stage.unlink(missing_ok=True)
        metadata_stage.unlink(missing_ok=True)
        rejections_stage.unlink(missing_ok=True)
