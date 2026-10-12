import json
import os
from pathlib import Path
import shutil
import tempfile

from .files import atomic_json, sha256_file, text_hash
from .store import RunStore, SPLITS
from .tokenization import LocalTokenizer, training_prompt


def export_dataset(
    store: RunStore, output_dir: Path, *, tokenizer_path: Path | None = None,
    context_limit: int | None = None,
) -> dict:
    if output_dir.exists():
        raise ValueError("Export directory already exists")
    status = store.status()
    if not status["generation_targets_met"]:
        raise ValueError("Accepted split targets are not met; no partial dataset will be exported")
    if status["accounting"]["uncertain_attempts"] or status["accounting"]["budget_exceeded"]:
        raise ValueError("Resolve billing uncertainty/overrun before exporting")
    approval = store.db.execute(
        "SELECT attempt_count,review_sha256 FROM approvals WHERE stage='export'"
    ).fetchone()
    if not approval or approval["attempt_count"] != status["accounting"]["attempts"]:
        raise ValueError("A current human-reviewed export approval is required")
    neutral = store.config.get("dataset_mode", "tokenizer-specific") == "model-neutral"
    if not neutral and (tokenizer_path is not None or context_limit is not None):
        raise ValueError("Tokenizer-specific runs export with their frozen tokenizer and context limit")
    if neutral and tokenizer_path is None and context_limit is not None:
        raise ValueError("A context limit requires an export tokenizer")
    if not neutral:
        tokenizer_path = store.run_dir / "tokenizer.json"
        context_limit = store.config["context_limit"]
    if tokenizer_path is not None:
        context_limit = 2048 if context_limit is None else context_limit
        if type(context_limit) is not int or context_limit <= 0:
            raise ValueError("Export context limit must be a positive integer")
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path is not None else None
    tokenizer = LocalTokenizer(tokenizer_path) if tokenizer_path is not None else None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".", dir=output_dir.parent))
    counts = {split: 0 for split in SPLITS}
    try:
        for split in SPLITS:
            with (temporary / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as pairs, (
                temporary / f"{split}.manifest.jsonl"
            ).open("w", encoding="utf-8", newline="\n") as provenance:
                for row in store.records(status="accepted", usable_only=True):
                    if row["split"] != split:
                        continue
                    document = row["record"]
                    summary = row["result"]["summary"]
                    tokens = None
                    if tokenizer is None:
                        pair = {"id": row["id"], "source": document["text"], "summary": summary}
                    else:
                        prompt = training_prompt(document["text"])
                        tokens = tokenizer.count(prompt + summary)
                        if tokens > context_limit:
                            raise ValueError(
                                f"Document {row['id']} exceeds the training context after generation; "
                                "export aborted, no truncation or silent removal"
                            )
                        pair = {"prompt": prompt, "completion": summary}
                    pairs.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    attempts = [
                        {**dict(attempt), "response": json.loads(attempt["response"]) if attempt["response"] else None}
                        for attempt in store.db.execute(
                            "SELECT id,kind,status,reserved_usd,cost_usd,response,reason,started_at,finished_at "
                            "FROM attempts WHERE document_id=? ORDER BY id", (row["id"],)
                        )
                    ]
                    manifest = {
                        "row": counts[split], **{key: value for key, value in document.items() if key != "text"},
                        "training_tokens": tokens, "prompt_version": store.config["prompt_version"],
                        "prompt_sha256": store.config["prompt_sha256"],
                        "schema_sha256": store.config["schema_sha256"],
                        "model": store.config["model"], "allowed_providers": store.config["providers"],
                        "attempts": attempts,
                    }
                    provenance.write(json.dumps(manifest, ensure_ascii=False) + "\n")
                    counts[split] += 1
                pairs.flush()
                provenance.flush()
                os.fsync(pairs.fileno())
                os.fsync(provenance.fileno())
        if counts != store.config["targets"]:
            raise ValueError("Accepted counts differ from exact split targets; export aborted")
        with (temporary / "outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as outcomes:
            for row in store.records():
                outcomes.write(json.dumps(row, ensure_ascii=False) + "\n")
            outcomes.flush()
            os.fsync(outcomes.fileno())
        with (temporary / "attempts.jsonl").open("w", encoding="utf-8", newline="\n") as attempts:
            for row in store.db.execute("SELECT * FROM attempts ORDER BY id"):
                value = dict(row)
                value["response"] = json.loads(value["response"]) if value["response"] else None
                attempts.write(json.dumps(value, ensure_ascii=False) + "\n")
            attempts.flush()
            os.fsync(attempts.fileno())
        with (temporary / "quality-holds.jsonl").open("w", encoding="utf-8", newline="\n") as holds:
            for row in store.db.execute("SELECT * FROM quality_holds ORDER BY document_id"):
                value = dict(row)
                value["details"] = json.loads(value["details"])
                holds.write(json.dumps(value, ensure_ascii=False) + "\n")
            holds.flush()
            os.fsync(holds.fileno())
        if store.collection_policy is not None:
            atomic_json(temporary / "collection-policy.json", store.collection_policy)
            atomic_json(temporary / "collection-stages.json", store._collection_stages())
            atomic_json(temporary / "collection-policy-amendments.json", store.collection_policy_history())
        extension = store._optional_metadata("candidate_extension")
        if extension is not None:
            atomic_json(temporary / "candidate-extension.json", extension)
        if status["accounting"]["assessment_attempts"]:
            with (temporary / "assessments.jsonl").open("w", encoding="utf-8", newline="\n") as assessments:
                for row in store.db.execute("SELECT * FROM assessments ORDER BY document_id"):
                    value = dict(row)
                    value["result"] = json.loads(value["result"]) if value["result"] else None
                    assessments.write(json.dumps(value, ensure_ascii=False) + "\n")
                assessments.flush()
                os.fsync(assessments.fileno())
            atomic_json(temporary / "assessment-protocol.json", store.metadata("assessment_protocol"))
        review_path = store.run_dir / "approved-export.json"
        if sha256_file(review_path) != approval["review_sha256"]:
            raise ValueError("Approved review file changed; export aborted")
        shutil.copyfile(review_path, temporary / "review.json")
        if tokenizer_path is not None and sha256_file(tokenizer_path) != tokenizer_hash:
            raise ValueError("Export tokenizer changed while writing; export aborted")
        report = {
            "format_version": 1,
            "format": "source-summary-jsonl" if tokenizer is None else "prompt-completion-jsonl",
            "counts": counts, "context_limit": context_limit,
            "tokenizer_sha256": tokenizer_hash,
            "training_prompt_sha256": text_hash(training_prompt("")) if tokenizer is not None else None,
            "length_contract": (
                "Not tokenized; downstream consumers must check their own template and context window."
                if tokenizer is None else
                "tokenizer.encode(prompt + completion, add_special_tokens=True); no truncation/padding"
            ),
            "alloy_loader_compatibility": "not_verified",
            "config": store.config, "accounting": status["accounting"],
            "review_sha256": approval["review_sha256"],
            "files": {path.name: sha256_file(path) for path in sorted(temporary.iterdir())},
        }
        atomic_json(temporary / "manifest.json", report)
        os.replace(temporary, output_dir)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
