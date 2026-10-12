import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
import shutil
import sqlite3
import time

from .files import atomic_json, load_json, sha256_file, text_hash
from .retries import RetryRateLimited, retryable_reason
from .tokenization import LocalTokenizer, training_prompt
from .validation import validate_teacher


SPLITS = ("train", "validation", "test")
SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    id TEXT PRIMARY KEY, split TEXT NOT NULL, record TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
    result TEXT, reason TEXT
);
CREATE TABLE attempts (
    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
    kind TEXT NOT NULL DEFAULT 'generation',
    retry_started_at REAL,
    status TEXT NOT NULL, reserved_usd TEXT NOT NULL, cost_usd TEXT,
    response TEXT, intended_status TEXT, reason TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT
);
CREATE INDEX document_queue ON documents(status, split, id);
CREATE INDEX attempts_by_document ON attempts(document_id, id);
CREATE INDEX attempt_status ON attempts(status, id);
CREATE TABLE approvals (
    stage TEXT PRIMARY KEY, attempt_count INTEGER NOT NULL,
    review_sha256 TEXT NOT NULL, approved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE reviews (
    id TEXT PRIMARY KEY, attempt_count INTEGER NOT NULL, sample TEXT NOT NULL
);
"""

ASSESSMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    document_id TEXT PRIMARY KEY REFERENCES documents(id),
    source_attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0, result TEXT, reason TEXT
)
"""

COLLECTION_TABLES = (
    """CREATE TABLE IF NOT EXISTS quality_holds (
        document_id TEXT PRIMARY KEY REFERENCES documents(id),
        reason TEXT NOT NULL, details TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS collection_stages (
        generation_attempts INTEGER NOT NULL, is_final INTEGER NOT NULL,
        result TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (generation_attempts, is_final)
    )""",
    """CREATE TABLE IF NOT EXISTS collection_policy_amendments (
        id INTEGER PRIMARY KEY, previous_policy TEXT NOT NULL, new_policy TEXT NOT NULL,
        superseded_stages TEXT NOT NULL, reason TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
)


def money(value: object, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("USD values must be finite decimal numbers")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("USD values must be finite decimal numbers") from None
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError("USD values must be finite and nonnegative (positive for limits)")
    return amount


def initialize_run(
    prepared: Path, run_dir: Path, *, model: str, providers: list[str],
    budget_usd: str, input_price_per_million: str, output_price_per_million: str,
    prompt_path: Path, schema_path: Path, tokenizer_path: Path | None = None,
    model_neutral: bool = False, target_train: int = 10000,
    target_validation: int = 200, target_test: int = 200, max_tokens: int = 2048,
    max_attempts: int = 3, context_limit: int = 2048, summary_token_reserve: int = 256,
) -> dict:
    if run_dir.exists():
        raise ValueError("Run directory already exists; use generate to resume")
    if model_neutral == (tokenizer_path is not None):
        raise ValueError("Choose either model-neutral mode or a local training tokenizer")
    if not model.strip() or not providers or any(not p.strip() for p in providers):
        raise ValueError("An explicit model and at least one provider are required")
    targets = dict(zip(SPLITS, (target_train, target_validation, target_test)))
    if any(type(count) is not int or count <= 0 for count in targets.values()):
        raise ValueError("All split targets must be positive integers")
    if not 256 <= max_tokens <= 16384 or not 1 <= max_attempts <= 5:
        raise ValueError("max_tokens must be 256..16384 and max_attempts 1..5")
    if not 0 < summary_token_reserve < context_limit:
        raise ValueError("Summary reserve must be positive and below the context limit")
    tokenizer = LocalTokenizer(tokenizer_path) if tokenizer_path is not None else None
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = load_json(schema_path)
    expected = {"accepted": "boolean", "summary": "string", "rejection_reason": "string"}
    if (
        schema.get("type") != "object"
        or not isinstance(schema.get("required"), list)
        or any(not isinstance(key, str) for key in schema["required"])
        or set(schema.get("required", [])) != set(expected)
        or schema.get("additionalProperties") is not False
        or not isinstance(schema.get("properties"), dict)
        or set(schema["properties"]) != set(expected)
        or any(schema["properties"][key] != {"type": value} for key, value in expected.items())
    ):
        raise ValueError("Schema must describe the supported teacher response contract")
    if not prompt.strip():
        raise ValueError("Teacher prompt must not be empty")
    document_path = prepared / "documents.jsonl"
    manifest = load_json(prepared / "manifest.json")
    documents_hash = sha256_file(document_path)
    if manifest.get("format_version") != 1 or manifest.get("documents_sha256") != documents_hash:
        raise ValueError("Prepared documents do not match their versioned manifest")
    preparation = manifest.get("arguments", {})
    if not isinstance(preparation, dict):
        raise ValueError("Invalid preparation parameters in manifest")
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path is not None else None
    if preparation.get("tokenizer_sha256") is not None and (
        preparation["tokenizer_sha256"] != tokenizer_hash
        or preparation.get("context_limit") != context_limit
        or preparation.get("summary_token_reserve") != summary_token_reserve
    ):
        raise ValueError("Use the tokenizer/context/reserve from preparation, or prepare a new candidate pool")
    config = {
        "format_version": 1, "model": model, "providers": providers,
        "budget_usd": str(money(budget_usd, positive=True)),
        "input_price_per_million": str(money(input_price_per_million, positive=True)),
        "output_price_per_million": str(money(output_price_per_million, positive=True)),
        "targets": targets, "max_tokens": max_tokens, "max_attempts": max_attempts,
        "reasoning_effort": "low", "prompt_version": prompt_path.name,
        "prompt_sha256": text_hash(prompt), "schema_sha256": text_hash(
            json.dumps(schema, sort_keys=True)
        ),
        "documents_sha256": documents_hash,
        "dataset_mode": "model-neutral" if model_neutral else "tokenizer-specific",
        "tokenizer_sha256": tokenizer_hash,
        "training_prompt_sha256": None if model_neutral else text_hash(training_prompt("")),
        "context_limit": None if model_neutral else context_limit,
        "summary_token_reserve": None if model_neutral else summary_token_reserve,
        "preparation_manifest": manifest,
    }
    run_dir.mkdir(parents=True)
    database = run_dir / "state.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SCHEMA)
        with connection:
            for key, value in (("config", config), ("prompt", prompt), ("schema", schema)):
                connection.execute(
                    "INSERT INTO metadata VALUES (?, ?)", (key, json.dumps(value))
                )
            with document_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid prepared JSON at line {line_number}") from None
                    required = (
                        "text", "source_id", "source_url", "source_dataset",
                        "source_revision", "source_license", "document_sha256", "split",
                    )
                    if not isinstance(record, dict) or any(
                        not isinstance(record.get(key), str) or not record[key].strip()
                        for key in required
                    ):
                        raise ValueError(f"Missing prepared provenance at line {line_number}")
                    if record["split"] not in SPLITS or text_hash(record["text"]) != record["document_sha256"]:
                        raise ValueError(f"Invalid prepared split or hash at line {line_number}")
                    if tokenizer is not None and (
                        tokenizer.count(training_prompt(record["text"])) + summary_token_reserve > context_limit
                    ):
                        raise ValueError(
                            f"Prepared line {line_number} exceeds source token allowance. "
                            "Prepare again with a smaller max_words BEFORE splitting; do not truncate."
                        )
                    try:
                        connection.execute(
                            "INSERT INTO documents (id, split, record) VALUES (?, ?, ?)",
                            (record["document_sha256"], record["split"], json.dumps(record)),
                        )
                    except sqlite3.IntegrityError:
                        raise ValueError(f"Duplicate prepared document at line {line_number}") from None
            counts = dict(connection.execute("SELECT split, COUNT(*) FROM documents GROUP BY split"))
            if any(counts.get(split, 0) < target for split, target in targets.items()):
                raise ValueError(
                    "Prepared candidate counts are below requested targets; prepare more candidates "
                    "or lower targets. Extra candidates are needed for rejections."
                )
            if sha256_file(document_path) != config["documents_sha256"]:
                raise ValueError("Prepared input changed during initialization")
        if tokenizer_path is not None:
            shutil.copyfile(tokenizer_path, run_dir / "tokenizer.json")
            if sha256_file(run_dir / "tokenizer.json") != config["tokenizer_sha256"]:
                raise ValueError("Tokenizer changed during initialization")
        atomic_json(run_dir / "config.json", config)
    except (ValueError, OSError, sqlite3.Error, UnicodeError):
        connection.close()
        database.unlink(missing_ok=True)
        for name in ("state.sqlite3-journal", "state.sqlite3-wal", "state.sqlite3-shm"):
            (run_dir / name).unlink(missing_ok=True)
        (run_dir / "tokenizer.json").unlink(missing_ok=True)
        if not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise
    finally:
        connection.close()
    with RunStore(run_dir) as store:
        return store.status()


class RunStore:
    def __init__(self, run_dir: Path, *, read_only: bool = False):
        run_dir = run_dir.resolve()
        self.run_dir = run_dir
        database = run_dir / "state.sqlite3"
        if not database.is_file():
            raise ValueError("Run does not exist; initialize it first")
        mode = "ro" if read_only else "rw"
        self.db = sqlite3.connect(database.as_uri() + f"?mode={mode}", uri=True, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        if read_only:
            self.db.execute("PRAGMA query_only = ON")
        else:
            self.db.execute("PRAGMA journal_mode = WAL")
            self.db.execute("PRAGMA synchronous = FULL")
        try:
            self.config = self.metadata("config")
            self.prompt = self.metadata("prompt")
            self.schema = self.metadata("schema")
            if self.config != load_json(run_dir / "config.json"):
                raise ValueError("Run configuration changed; create a new run instead")
            if self.config.get("dataset_mode", "tokenizer-specific") == "tokenizer-specific":
                if sha256_file(run_dir / "tokenizer.json") != self.config["tokenizer_sha256"]:
                    raise ValueError("Run tokenizer changed; create a new run instead")
                if text_hash(training_prompt("")) != self.config["training_prompt_sha256"]:
                    raise ValueError("Training prompt template changed; create a new run instead")
            elif self.config["dataset_mode"] != "model-neutral":
                raise ValueError("Unknown dataset mode")
            if not read_only:
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    columns = {row["name"] for row in self.db.execute("PRAGMA table_info(attempts)")}
                    if "kind" not in columns:
                        self.db.execute("ALTER TABLE attempts ADD COLUMN kind TEXT NOT NULL DEFAULT 'generation'")
                    if "retry_started_at" not in columns:
                        self.db.execute("ALTER TABLE attempts ADD COLUMN retry_started_at REAL")
                    self.db.execute("CREATE INDEX IF NOT EXISTS attempt_retry_time ON attempts(retry_started_at)")
                    self.db.execute(ASSESSMENT_SCHEMA)
                    for statement in COLLECTION_TABLES:
                        self.db.execute(statement)
            self._apply_collection_policy()
            self._apply_retry_policy()
        except (ValueError, OSError, sqlite3.Error):
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def metadata(self, key: str):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Incomplete run metadata")
        return json.loads(row["value"])

    def _optional_metadata(self, key: str):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def _apply_collection_policy(self) -> None:
        self.collection_policy = self._optional_metadata("collection_policy")
        if self.collection_policy is not None:
            policy = self.collection_policy
            if policy["model"] != self.config["model"] or policy["providers"] != self.config["providers"]:
                raise ValueError("Collection authorization does not match the frozen model/provider")
            self.config = {
                **self.config, "budget_usd": policy["budget_usd"],
                "targets": policy["targets"], "collection_policy": policy,
            }

    def _apply_retry_policy(self) -> None:
        self.retry_policy = self._optional_metadata("retry_policy")
        if self.retry_policy is not None:
            self.config = {
                **self.config, "max_attempts": self.retry_policy["max_retries_per_sample"] + 1,
                "retry_policy": self.retry_policy,
            }

    def accounting(self) -> dict:
        known = Decimal(0)
        reserved = Decimal(0)
        unknown = 0
        attempts = 0
        generation_attempts = 0
        assessment_attempts = 0
        blocking_unknown = 0
        unconfirmed = 0
        retries = 0
        for row in self.db.execute("SELECT cost_usd, reserved_usd, kind, status,retry_started_at FROM attempts"):
            attempts += 1
            generation_attempts += row["kind"] == "generation"
            assessment_attempts += row["kind"] == "assessment"
            retries += row["retry_started_at"] is not None
            if row["cost_usd"] is None:
                reserved += money(row["reserved_usd"])
                unknown += 1
                blocking_unknown += row["status"] != "billing_unconfirmed"
                unconfirmed += row["status"] == "billing_unconfirmed"
            else:
                known += money(row["cost_usd"])
        budget = money(self.config["budget_usd"])
        return {
            "known_cost_usd": str(known), "uncertain_reserved_usd": str(reserved),
            "committed_usd": str(known + reserved), "budget_usd": str(budget),
            "remaining_usd": str(budget - known - reserved),
            "uncertain_attempts": unknown, "attempts": attempts,
            "blocking_uncertain_attempts": blocking_unknown,
            "unconfirmed_billing_attempts": unconfirmed,
            "retry_attempts": retries,
            "generation_attempts": generation_attempts, "assessment_attempts": assessment_attempts,
            "budget_exceeded": known + reserved > budget,
        }

    def status(self) -> dict:
        counts = {split: {} for split in SPLITS}
        for row in self.db.execute("SELECT split, status, COUNT(*) AS n FROM documents GROUP BY split, status"):
            counts[row["split"]][row["status"]] = row["n"]
        accounting = self.accounting()
        targets = self.config["targets"]
        usable = self.usable_counts()
        held = dict(self.db.execute(
            "SELECT d.split,COUNT(*) FROM quality_holds h JOIN documents d ON d.id=h.document_id "
            "WHERE d.status='accepted' GROUP BY d.split"
        ))
        complete = all(usable[split] >= targets[split] for split in SPLITS)
        return {
            "model": self.config["model"], "providers": self.config["providers"],
            "dataset_mode": self.config.get("dataset_mode", "tokenizer-specific"),
            "targets": targets, "counts": counts, "accounting": accounting,
            "usable_counts": usable, "quarantined_counts": {split: held.get(split, 0) for split in SPLITS},
            "generation_targets_met": complete,
            "approvals": [row["stage"] for row in self.db.execute("SELECT stage FROM approvals ORDER BY stage")],
            "inflight_attempts": [
                row["id"] for row in self.db.execute("SELECT id FROM attempts WHERE status='inflight'")
            ],
            "uncertain_attempt_ids": [
                row["id"] for row in self.db.execute("SELECT id FROM attempts WHERE status='uncertain'")
            ],
            "reservation_overruns": [
                row["id"] for row in self.db.execute(
                    "SELECT id,cost_usd,reserved_usd FROM attempts WHERE cost_usd IS NOT NULL"
                ) if money(row["cost_usd"]) > money(row["reserved_usd"])
            ],
            "unconfirmed_billing_attempt_ids": [
                row["id"] for row in self.db.execute("SELECT id FROM attempts WHERE status='billing_unconfirmed'")
            ],
            "retry_policy": self.retry_policy,
        }

    def next_document(self) -> dict | None:
        counts = self.usable_counts()
        eligible = [
            split for split in SPLITS
            if counts.get(split, 0) < self.config["targets"][split]
        ]
        if not eligible:
            return None
        # Proportional progress brings held-out documents into early review batches.
        eligible.sort(key=lambda s: counts.get(s, 0) / self.config["targets"][s])
        for split in eligible:
            row = self.db.execute(
                "SELECT record FROM documents WHERE status='pending' AND split=? ORDER BY id LIMIT 1",
                (split,),
            ).fetchone()
            if row:
                return json.loads(row["record"])
        return None

    def reserve(self, document_id: str, amount: Decimal) -> int:
        return self._reserve(document_id, amount, kind="generation")

    def reserve_assessment(self, document_id: str, amount: Decimal) -> int:
        return self._reserve(document_id, amount, kind="assessment")

    def _reserve(self, document_id: str, amount: Decimal, *, kind: str) -> int:
        amount = money(amount, positive=True)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._apply_retry_policy()
            if self.db.execute(
                "SELECT 1 FROM attempts WHERE status IN ('inflight','uncertain') LIMIT 1"
            ).fetchone():
                raise ValueError("An in-flight or uncertain request needs recovery/reconciliation first")
            if any(
                money(row["cost_usd"]) > money(row["reserved_usd"])
                for row in self.db.execute("SELECT cost_usd,reserved_usd FROM attempts WHERE cost_usd IS NOT NULL")
            ):
                raise ValueError("A provider charge exceeded its reservation; investigate before creating a new run")
            accounting = self.accounting()
            limit = self.db.execute("SELECT value FROM metadata WHERE key='request_authorization'").fetchone()
            if limit is not None and accounting["attempts"] >= json.loads(limit["value"])["max_total_requests"]:
                raise ValueError("Approved total request limit reached; no request sent")
            if kind == "generation":
                if self.collection_policy is not None:
                    stages = self._collection_stages()
                    if any(not stage["passed"] for stage in stages):
                        raise ValueError("Automated collection gate failed; collection is paused")
                    required = self._required_checkpoint(accounting["generation_attempts"], stages)
                    if required is not None:
                        raise ValueError(f"Automated collection gate at {required} requests must pass first")
                else:
                    approvals = {row["stage"] for row in self.db.execute("SELECT stage FROM approvals")}
                    if accounting["generation_attempts"] >= 1000 and "full" not in approvals:
                        raise ValueError("Review gate: approve full stage after reviewing the 1,000-request pilot")
                    if accounting["generation_attempts"] >= 100 and "scale" not in approvals:
                        raise ValueError("Review gate: approve scale stage after reviewing the 100-request pilot")
            if amount > Decimal(accounting["remaining_usd"]):
                raise ValueError("Budget cannot cover the next worst-case reservation; no request sent")
            table, id_column = ("documents", "id") if kind == "generation" else ("assessments", "document_id")
            row = self.db.execute(f"SELECT status, attempts FROM {table} WHERE {id_column}=?", (document_id,)).fetchone()
            if row is None or row["status"] != "pending" or row["attempts"] >= self.config["max_attempts"]:
                raise ValueError("Document is not eligible for another attempt")
            retry_started_at = None
            if self.retry_policy is not None and row["attempts"] > 0:
                if not self._retry_available(document_id, kind):
                    raise ValueError("Sample retry limit reached")
                now = time.time()
                previous = self.db.execute("SELECT MAX(retry_started_at) FROM attempts").fetchone()[0]
                if previous is not None:
                    wait = previous + self.retry_policy["min_retry_interval_seconds"] - now
                    if wait > 0:
                        raise RetryRateLimited(wait)
                retry_started_at = now
            cursor = self.db.execute(
                "INSERT INTO attempts (document_id,kind,status,reserved_usd,retry_started_at) "
                "VALUES (?,?,'inflight',?,?)",
                (document_id, kind, str(amount), retry_started_at),
            )
            self.db.execute(
                f"UPDATE {table} SET status='inflight', attempts=attempts+1 WHERE {id_column}=?", (document_id,)
            )
            self.db.commit()
            return cursor.lastrowid
        except (ValueError, sqlite3.Error):
            self.db.rollback()
            raise

    def finish(
        self, attempt_id: int, *, status: str, cost: Decimal | None,
        response: dict | None = None, result: dict | None = None, reason: str = "",
    ) -> None:
        self._finish(
            attempt_id, status=status, cost=cost, response=response, result=result, reason=reason, kind="generation"
        )

    def finish_assessment(
        self, attempt_id: int, *, status: str, cost: Decimal | None,
        response: dict | None = None, result: dict | None = None, reason: str = "",
    ) -> None:
        self._finish(
            attempt_id, status=status, cost=cost, response=response, result=result, reason=reason, kind="assessment"
        )

    def _finish(
        self, attempt_id: int, *, status: str, cost: Decimal | None,
        response: dict | None, result: dict | None, reason: str, kind: str,
    ) -> None:
        valid = (
            ("accepted", "rejected", "flagged", "error", "pending", "uncertain")
            if kind == "generation" else ("completed", "error", "pending", "uncertain")
        )
        if status not in valid:
            raise ValueError("Unknown completion status")
        cost = None if cost is None else money(cost)
        with self.db:
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None or row["status"] != "inflight":
                raise ValueError("Attempt is no longer in flight")
            if row["kind"] != kind:
                raise ValueError("Completion kind does not match the reserved attempt")
            uncertain = cost is None
            effective_status = status
            effective_reason = reason
            unconfirmed = uncertain and self.retry_policy is not None and status != "uncertain"
            if self.retry_policy is not None and status == "error" and (
                response is not None or retryable_reason(reason)
            ):
                if self._retry_available(row["document_id"], kind):
                    effective_status = "pending"
                    effective_reason = "retry_pending:" + reason
                else:
                    effective_reason = "retry_exhausted:" + reason
            self.db.execute(
                "UPDATE attempts SET status=?, cost_usd=?, response=?, intended_status=?, "
                "reason=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                ("billing_unconfirmed" if unconfirmed else "uncertain" if uncertain else "completed",
                 None if uncertain else str(cost),
                 json.dumps(response) if response is not None else None, status, reason, attempt_id),
            )
            table, id_column = ("documents", "id") if kind == "generation" else ("assessments", "document_id")
            self.db.execute(
                f"UPDATE {table} SET status=?, result=?, reason=? WHERE {id_column}=?",
                ("uncertain" if uncertain and not unconfirmed else effective_status,
                 json.dumps(result) if result is not None else None, effective_reason, row["document_id"]),
            )

    def recover(self, attempt_id: int) -> None:
        row = self.db.execute("SELECT kind FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown attempt ID")
        finish = self.finish_assessment if row["kind"] == "assessment" else self.finish
        finish(attempt_id, status="uncertain", cost=None, reason="interrupted_request")

    def reconcile(self, attempt_id: int, cost: Decimal, *, retry: bool = False) -> None:
        cost = money(cost)
        with self.db:
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None or row["status"] not in ("uncertain", "billing_unconfirmed"):
                raise ValueError("Only uncertain attempts can be reconciled")
            if row["status"] == "billing_unconfirmed":
                if retry:
                    raise ValueError("This request has already been handled by the retry policy; reconcile charge only")
                self.db.execute(
                    "UPDATE attempts SET status='reconciled',cost_usd=? WHERE id=?", (str(cost), attempt_id)
                )
                return
            intended = row["intended_status"]
            assessment = row["kind"] == "assessment"
            table, id_column = ("assessments", "document_id") if assessment else ("documents", "id")
            document = self.db.execute(
                f"SELECT attempts FROM {table} WHERE {id_column}=?", (row["document_id"],)
            ).fetchone()
            saved_response = ("completed",) if assessment else ("accepted", "rejected", "flagged")
            if retry and intended in saved_response:
                raise ValueError("A response already exists; reconcile without retry")
            if retry and document["attempts"] >= self.config["max_attempts"]:
                raise ValueError("Document retry limit reached")
            status = "pending" if retry else (
                intended if intended in (*saved_response, "error") else "error"
            )
            self.db.execute(
                "UPDATE attempts SET status='reconciled', cost_usd=? WHERE id=?", (str(cost), attempt_id)
            )
            self.db.execute(
                f"UPDATE {table} SET status=?, reason=CASE WHEN ?='error' THEN "
                f"'reconciled_without_usable_response' ELSE reason END WHERE {id_column}=?",
                (status, status, row["document_id"]),
            )

    def authorize_request_limit(self, max_total_requests: int, *, budget_usd: Decimal, acknowledge: bool) -> None:
        if not acknowledge:
            raise ValueError("Changing the total request ceiling requires explicit paid authorization")
        if type(max_total_requests) is not int or max_total_requests <= 0:
            raise ValueError("Total request ceiling must be a positive integer")
        if money(budget_usd, positive=True) != money(self.config["budget_usd"]):
            raise ValueError("Authorization must match the frozen total run budget, including prior charges")
        if max_total_requests < self.accounting()["attempts"]:
            raise ValueError("Request ceiling cannot be below attempts already made")
        authorization = {
            "max_total_requests": max_total_requests, "budget_usd": str(budget_usd),
            "model": self.config["model"], "providers": self.config["providers"],
            "human_review_approval": False,
        }
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadata (key,value) VALUES ('request_authorization',?)",
                (json.dumps(authorization),),
            )
        atomic_json(self.run_dir / "request-authorization.json", authorization)

    def freeze_assessment(self, prompt: str, schema: dict, prompt_version: str) -> None:
        if not prompt.strip() or not isinstance(schema, dict) or not prompt_version.strip():
            raise ValueError("A versioned assessment prompt and schema are required")
        protocol = {
            "prompt": prompt, "schema": schema, "prompt_version": prompt_version,
            "prompt_sha256": text_hash(prompt),
            "schema_sha256": text_hash(json.dumps(schema, sort_keys=True)),
            "model": self.config["model"], "providers": self.config["providers"],
            "independent_verification": False,
        }
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.accounting()["blocking_uncertain_attempts"]:
                raise ValueError("Resolve in-flight or uncertain billing before assessment")
            saved = self.db.execute("SELECT value FROM metadata WHERE key='assessment_protocol'").fetchone()
            if saved is not None:
                if json.loads(saved["value"]) != protocol:
                    raise ValueError("Assessment protocol changed; resume with the original prompt/schema")
                return
            self.db.execute(
                "INSERT INTO metadata (key,value) VALUES ('assessment_protocol',?)", (json.dumps(protocol),)
            )
            self.db.execute(
                "INSERT INTO assessments (document_id, source_attempt_id) "
                "SELECT d.id, MAX(a.id) FROM documents d JOIN attempts a ON a.document_id=d.id "
                "WHERE d.status IN ('accepted','flagged','rejected','error') AND a.kind='generation' "
                "GROUP BY d.id"
            )
            if not self.db.execute("SELECT 1 FROM assessments LIMIT 1").fetchone():
                raise ValueError("No completed generation outcomes to assess")
        atomic_json(self.run_dir / "assessment-protocol.json", protocol)

    def next_assessment(self) -> dict | None:
        row = self.db.execute(
            "SELECT d.*, a.response FROM assessments s JOIN documents d ON d.id=s.document_id "
            "JOIN attempts a ON a.id=s.source_attempt_id WHERE s.status='pending' ORDER BY d.id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        for key in ("record", "result", "response"):
            value[key] = json.loads(value[key]) if value[key] else None
        return value

    def assessment_attempt_count(self, document_id: str) -> int:
        row = self.db.execute("SELECT attempts FROM assessments WHERE document_id=?", (document_id,)).fetchone()
        if row is None:
            raise ValueError("Document is not part of the frozen assessment sample")
        return row["attempts"]

    def assessment_status(self) -> dict:
        counts = dict(self.db.execute("SELECT status,COUNT(*) FROM assessments GROUP BY status"))
        verdicts = {}
        for row in self.db.execute("SELECT result FROM assessments WHERE status='completed'"):
            result = json.loads(row["result"])
            verdict = result["verdict"]
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        return {
            "model": self.config["model"], "providers": self.config["providers"],
            "counts": counts, "verdicts": verdicts, "accounting": self.accounting(),
            "independent_verification": False, "human_approval_granted": False,
        }

    def _retry_available(self, document_id: str, kind: str) -> bool:
        if self.retry_policy is None:
            return False
        table, column = ("documents", "id") if kind == "generation" else ("assessments", "document_id")
        row = self.db.execute(
            f"SELECT attempts FROM {table} WHERE {column}=?", (document_id,)
        ).fetchone()
        retries_used = self.db.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT kind) FROM attempts WHERE document_id=?", (document_id,)
        ).fetchone()[0]
        return (
            row is not None and row["attempts"] < self.config["max_attempts"]
            and retries_used < self.retry_policy["max_retries_per_sample"]
        )

    def authorize_retry_policy(
        self, *, max_retries_per_sample: int = 3, min_retry_interval_seconds: int = 60,
        acknowledge: bool,
    ) -> dict:
        if not acknowledge:
            raise ValueError("Continuing with uncertain charges and bounded retries requires explicit approval")
        if type(max_retries_per_sample) is not int or not 0 <= max_retries_per_sample <= 3:
            raise ValueError("Retries per sample must be between zero and three")
        if type(min_retry_interval_seconds) is not int or min_retry_interval_seconds < 60:
            raise ValueError("The global retry interval must be at least 60 seconds")
        policy = {
            "max_retries_per_sample": max_retries_per_sample,
            "max_attempts_per_operation": max_retries_per_sample + 1,
            "min_retry_interval_seconds": min_retry_interval_seconds,
            "scope": "one shared retry allowance per source across generation and assessment",
            "continue_with_reserved_uncertain_charges": True,
            "skip_exhausted_samples": True,
        }
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM attempts WHERE status='inflight' LIMIT 1").fetchone():
                raise ValueError("Stop the active worker before changing retry policy")
            saved = self._optional_metadata("retry_policy")
            if saved is not None and saved != policy:
                raise ValueError("Retry policy is already frozen; retry history cannot be reset")
            self.db.execute(
                "INSERT OR IGNORE INTO metadata (key,value) VALUES ('retry_policy',?)", (json.dumps(policy),)
            )
            self.db.execute(
                "UPDATE attempts SET retry_started_at=CAST(strftime('%s',started_at) AS REAL) "
                "WHERE retry_started_at IS NULL AND EXISTS "
                "(SELECT 1 FROM attempts older WHERE older.document_id=attempts.document_id "
                "AND older.kind=attempts.kind AND older.id<attempts.id)"
            )
        self._apply_retry_policy()
        requeued = 0
        unconfirmed = 0
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for row in self.db.execute("SELECT * FROM attempts WHERE status='uncertain' ORDER BY id").fetchall():
                self.db.execute("UPDATE attempts SET status='billing_unconfirmed' WHERE id=?", (row["id"],))
                unconfirmed += 1
                latest = self.db.execute(
                    "SELECT MAX(id) FROM attempts WHERE document_id=? AND kind=?",
                    (row["document_id"], row["kind"]),
                ).fetchone()[0]
                if latest != row["id"]:
                    continue
                table, column = (
                    ("documents", "id") if row["kind"] == "generation" else ("assessments", "document_id")
                )
                intended = row["intended_status"]
                success_states = ("accepted", "flagged", "rejected") if row["kind"] == "generation" else ("completed",)
                if intended in success_states:
                    status, reason = intended, row["reason"]
                else:
                    retryable = row["response"] is not None or retryable_reason(row["reason"] or "")
                    retry = retryable and self._retry_available(row["document_id"], row["kind"])
                    status = "pending" if retry else "error"
                    reason = ("retry_pending:" if retry else "retry_exhausted:") + (row["reason"] or "unknown_failure")
                    requeued += retry
                self.db.execute(
                    f"UPDATE {table} SET status=?,reason=? WHERE {column}=?",
                    (status, reason, row["document_id"]),
                )
        atomic_json(self.run_dir / "retry-policy.json", policy)
        return {
            "retry_policy": policy, "requeued_samples": requeued,
            "charges_retained_as_unconfirmed": unconfirmed, "accounting": self.accounting(),
        }

    def usable_counts(self) -> dict:
        counts = dict(self.db.execute(
            "SELECT d.split,COUNT(*) FROM documents d WHERE d.status='accepted' "
            "AND NOT EXISTS (SELECT 1 FROM quality_holds h WHERE h.document_id=d.id) GROUP BY d.split"
        ))
        return {split: counts.get(split, 0) for split in SPLITS}

    def quality_hold(self, document_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT reason,details FROM quality_holds WHERE document_id=?", (document_id,)
        ).fetchone()
        return {"reason": row["reason"], "details": json.loads(row["details"])} if row else None

    def authorize_collection(
        self, *, budget_usd: Decimal, target_train: int, target_validation: int, target_test: int,
        structural_threshold: Decimal = Decimal(".98"), usable_threshold: Decimal = Decimal(".75"),
        max_total_requests: int = 30000, acknowledge: bool,
    ) -> dict:
        if not acknowledge:
            raise ValueError("Automatic collection requires explicit budget, target and stage-policy authorization")
        targets = dict(zip(SPLITS, (target_train, target_validation, target_test)))
        if any(type(value) is not int or value <= 0 for value in targets.values()):
            raise ValueError("Collection split targets must be positive integers")
        budget = money(budget_usd, positive=True)
        structural = money(structural_threshold, positive=True)
        usable = money(usable_threshold, positive=True)
        if structural > 1 or usable > 1:
            raise ValueError("Collection thresholds must be fractions between zero and one")
        if type(max_total_requests) is not int or max_total_requests <= 0:
            raise ValueError("Collection request ceiling must be positive")
        policy = {
            "format_version": 1, "mode": "automatic_collection_unreviewed",
            "model": self.config["model"], "providers": self.config["providers"],
            "budget_usd": str(budget), "targets": targets,
            "structural_threshold": str(structural), "usable_threshold": str(usable),
            "structural_denominator": "model_responses",
            "max_total_requests": max_total_requests,
            "checkpoints": "100,1000,and_every_1000_generation_attempts_then_final",
            "human_review_approval": False, "publication_authorized": False,
        }
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            accounting = self.accounting()
            if accounting["blocking_uncertain_attempts"]:
                raise ValueError("Resolve uncertain billing before authorizing collection")
            if budget < money(accounting["committed_usd"]):
                raise ValueError("Collection budget is below charges already incurred")
            if max_total_requests < accounting["attempts"]:
                raise ValueError("Collection request limit is below attempts already incurred")
            existing = self._optional_metadata("collection_policy")
            if existing is not None and existing != policy:
                raise ValueError("Collection policy is already frozen; use an explicit audited amendment")
            available = dict(self.db.execute("SELECT split,COUNT(*) FROM documents GROUP BY split"))
            counts = self.usable_counts()
            if any(available.get(split, 0) < target or counts[split] > target for split, target in targets.items()):
                raise ValueError("Collection targets must cover current usable counts and fit the candidate pool")
            self.db.execute(
                "INSERT OR IGNORE INTO metadata (key,value) VALUES ('collection_policy',?)",
                (json.dumps(policy),),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO metadata (key,value) VALUES ('request_authorization',?)",
                (json.dumps({
                    "max_total_requests": max_total_requests, "budget_usd": str(budget),
                    "model": policy["model"], "providers": policy["providers"], "human_review_approval": False,
                }),),
            )
        self._apply_collection_policy()
        atomic_json(self.run_dir / "collection-policy.json", policy)
        self.quarantine_assessment_findings()
        return self.collection_status()

    def amend_usable_threshold(
        self, threshold: Decimal, *, reason: str, acknowledge: bool,
    ) -> dict:
        if not acknowledge:
            raise ValueError("Changing the collection threshold requires explicit approval")
        threshold = money(threshold, positive=True)
        if threshold > 1 or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Provide a threshold between zero and one and an amendment reason")
        if (self.run_dir / ".collection.lock").exists():
            raise ValueError("Stop the collector before amending its policy")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            previous = self._optional_metadata("collection_policy")
            if previous is None:
                raise ValueError("No automatic collection policy exists")
            if self.accounting()["blocking_uncertain_attempts"]:
                raise ValueError("Resolve active or blocking uncertain requests before amending policy")
            old_threshold = money(previous["usable_threshold"], positive=True)
            if threshold > old_threshold:
                raise ValueError("This amendment only lowers the usable threshold; other settings remain frozen")
            if threshold == old_threshold:
                return self.collection_status()
            revised = {**previous, "usable_threshold": str(threshold)}
            amendment_id = self._record_policy_amendment(previous, revised, reason.strip())
        return self._publish_policy_amendment(amendment_id)

    def _record_policy_amendment(self, previous: dict, revised: dict, reason: str) -> int:
        superseded = [
            {**dict(row), "result": json.loads(row["result"])}
            for row in self.db.execute("SELECT * FROM collection_stages ORDER BY generation_attempts,is_final")
            if not json.loads(row["result"])["passed"]
        ]
        cursor = self.db.execute(
            "INSERT INTO collection_policy_amendments "
            "(previous_policy,new_policy,superseded_stages,reason) VALUES (?,?,?,?)",
            (json.dumps(previous), json.dumps(revised), json.dumps(superseded), reason),
        )
        for stage in superseded:
            self.db.execute(
                "DELETE FROM collection_stages WHERE generation_attempts=? AND is_final=?",
                (stage["generation_attempts"], stage["is_final"]),
            )
        self.db.execute("UPDATE metadata SET value=? WHERE key='collection_policy'", (json.dumps(revised),))
        return cursor.lastrowid

    def _publish_policy_amendment(self, amendment_id: int) -> dict:
        self._apply_collection_policy()
        atomic_json(self.run_dir / "collection-policy.json", self.collection_policy)
        amendment = next(item for item in self.collection_policy_history() if item["id"] == amendment_id)
        atomic_json(self.run_dir / f"collection-policy-amendment-{amendment_id}.json", amendment)
        return self.collection_status()

    def correct_structural_metric(self, *, acknowledge: bool) -> dict:
        if not acknowledge:
            raise ValueError("Reclassifying historical checkpoint metrics requires explicit approval")
        if (self.run_dir / ".collection.lock").exists():
            raise ValueError("Stop the collector before correcting its checkpoint metric")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            previous = self._optional_metadata("collection_policy")
            if previous is None:
                raise ValueError("No automatic collection policy exists")
            if self.accounting()["blocking_uncertain_attempts"]:
                raise ValueError("Resolve active or blocking uncertain requests before correcting the metric")
            if previous.get("structural_denominator") == "model_responses":
                return self.collection_status()
            revised = {**previous, "structural_denominator": "model_responses"}
            amendment_id = self._record_policy_amendment(
                previous, revised,
                "Correct structural validity to use returned model completions, not transport failures. "
                "Rate limits remain in attempt counts, retry accounting, and usable-yield denominators. "
                "No threshold, budget, source outcome, or human approval changed.",
            )
        return self._publish_policy_amendment(amendment_id)

    def collection_policy_history(self) -> list[dict]:
        history = []
        for row in self.db.execute("SELECT * FROM collection_policy_amendments ORDER BY id"):
            item = dict(row)
            for key in ("previous_policy", "new_policy", "superseded_stages"):
                item[key] = json.loads(item[key])
            history.append(item)
        return history

    def quarantine_assessment_findings(self) -> int:
        inserted = 0
        with self.db:
            for row in self.db.execute(
                "SELECT document_id,status,result,reason FROM assessments WHERE status IN ('completed','error')"
            ).fetchall():
                result = json.loads(row["result"]) if row["result"] else None
                if row["status"] == "completed" and result and (
                    result.get("verdict") == "clear" and result.get("disposition_correct") is True
                ):
                    continue
                reason = "assessment_error" if row["status"] == "error" else "assessment_concern"
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO quality_holds (document_id,reason,details) VALUES (?,?,?)",
                    (row["document_id"], reason, json.dumps({
                        "assessment_status": row["status"], "result": result, "reason": row["reason"],
                        "independent_verification": False,
                    })),
                )
                inserted += cursor.rowcount
        return inserted

    def _collection_stages(self) -> list[dict]:
        return [
            json.loads(row["result"]) for row in self.db.execute(
                "SELECT result FROM collection_stages ORDER BY generation_attempts,is_final"
            )
        ]

    @staticmethod
    def _required_checkpoint(generation_attempts: int, stages: list[dict]) -> int | None:
        boundary = (
            (generation_attempts // 1000) * 1000 if generation_attempts >= 1000
            else 100 if generation_attempts >= 100 else None
        )
        if boundary is None:
            return None
        if any(
            stage["generation_attempts"] == boundary and not stage["final"] and stage["passed"]
            for stage in stages
        ):
            return None
        return boundary

    def collection_status(self) -> dict:
        status = self.status()
        generations = status["accounting"]["generation_attempts"]
        stages = self._collection_stages()
        return {
            **status, "collection_policy": self.collection_policy,
            "required_checkpoint": self._required_checkpoint(generations, stages),
            "next_checkpoint": 100 if generations < 100 else (generations // 1000 + 1) * 1000,
            "stages": stages, "failed_gate": any(not stage["passed"] for stage in stages),
            "policy_amendment_count": self.db.execute("SELECT COUNT(*) FROM collection_policy_amendments").fetchone()[0],
            "human_review_required_before_release": True,
        }

    def evaluate_collection_gate(self, *, final: bool = False) -> dict:
        if self.collection_policy is None:
            raise ValueError("No authorized automatic collection policy")
        self.quarantine_assessment_findings()
        status = self.collection_status()
        if status["accounting"]["blocking_uncertain_attempts"] or status["reservation_overruns"]:
            raise ValueError("Resolve accounting uncertainty or reservation overruns before continuing")
        if status["failed_gate"]:
            raise ValueError("A collection gate has already failed; collection remains paused")
        through = status["accounting"]["generation_attempts"] if final else status["required_checkpoint"]
        if through is None or through <= 0:
            raise ValueError("No completed collection checkpoint is ready")
        saved = self.db.execute(
            "SELECT result FROM collection_stages WHERE generation_attempts=? AND is_final=?",
            (through, int(final)),
        ).fetchone()
        if saved:
            return json.loads(saved["result"])
        earlier = [
            stage["generation_attempts"] for stage in status["stages"]
            if not stage["final"] and stage["generation_attempts"] < through
        ]
        start = max(earlier, default=0)
        attempts = self.db.execute(
            "SELECT a.document_id,a.response,d.status,h.document_id AS held FROM attempts a "
            "JOIN documents d ON d.id=a.document_id LEFT JOIN quality_holds h ON h.document_id=d.id "
            "WHERE a.kind='generation' ORDER BY a.id LIMIT ? OFFSET ?", (through - start, start)
        ).fetchall()
        structurally_valid = 0
        model_responses = 0
        usable_documents = set()
        for row in attempts:
            response = json.loads(row["response"]) if row["response"] else None
            if response is None:
                continue
            model_responses += 1
            try:
                parsed = validate_teacher(response.get("content"), response.get("finish_reason"))
            except ValueError:
                continue
            structurally_valid += 1
            if parsed.accepted and row["status"] == "accepted" and row["held"] is None:
                usable_documents.add(row["document_id"])
        attempted = len(attempts)
        structural_denominator = self.collection_policy.get("structural_denominator", "all_attempts")
        if structural_denominator not in ("model_responses", "all_attempts"):
            raise ValueError("Unsupported structural gate denominator")
        structure_total = model_responses if structural_denominator == "model_responses" else attempted
        structure_fraction = Decimal(structurally_valid) / structure_total if structure_total else Decimal(0)
        usable_fraction = Decimal(len(usable_documents)) / attempted if attempted else Decimal(0)
        result = {
            "generation_attempts": through, "window_start": start, "attempted": attempted,
            "structurally_valid": structurally_valid, "usable": len(usable_documents),
            "model_responses": model_responses, "transport_failures": attempted - model_responses,
            "structural_denominator": structural_denominator,
            "structural_fraction": str(structure_fraction), "usable_fraction": str(usable_fraction),
            "structural_threshold": self.collection_policy["structural_threshold"],
            "usable_threshold": self.collection_policy["usable_threshold"],
            "final": final,
            "passed": (
                attempted > 0
                and structure_fraction >= money(self.collection_policy["structural_threshold"])
                and usable_fraction >= money(self.collection_policy["usable_threshold"])
                and not status["accounting"]["budget_exceeded"]
                and (not final or status["generation_targets_met"])
            ),
            "human_review_approval": False, "not_a_factual_accuracy_measurement": True,
        }
        with self.db:
            self.db.execute(
                "INSERT INTO collection_stages (generation_attempts,is_final,result) VALUES (?,?,?)",
                (through, int(final), json.dumps(result)),
            )
        atomic_json(self.run_dir / f"collection-stage-{through}{'-final' if final else ''}.json", result)
        return result

    def extend_candidates(self, prepared: Path, *, acknowledge: bool) -> dict:
        if not acknowledge:
            raise ValueError("Extending the frozen candidate pool requires explicit authorization")
        manifest = load_json(prepared / "manifest.json")
        path = prepared / "documents.jsonl"
        checksum = sha256_file(path)
        if manifest.get("format_version") != 1 or manifest.get("documents_sha256") != checksum:
            raise ValueError("Expanded candidates do not match their manifest")
        previous = self._optional_metadata("candidate_extension")
        current_hash = previous["documents_sha256"] if previous else self.config["documents_sha256"]
        if manifest.get("preserved_documents_sha256") != current_hash:
            raise ValueError("Expanded preparation must preserve the current candidate snapshot")
        existing = {
            row["id"]: json.loads(row["record"]) for row in self.db.execute("SELECT id,record FROM documents")
        }
        identities = {
            (row["source_dataset"], row["source_revision"], row["source_id"]): key
            for key, row in existing.items()
        }
        allowed_origins = {
            (row["source_dataset"], row["source_revision"], row["source_license"]) for row in existing.values()
        }
        tokenizer = (
            LocalTokenizer(self.run_dir / "tokenizer.json")
            if self.config.get("dataset_mode", "tokenizer-specific") == "tokenizer-specific" else None
        )
        seen = set()
        preserved = set()
        added = 0
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.accounting()["blocking_uncertain_attempts"]:
                raise ValueError("Stop all workers and resolve billing uncertainty before extending sources")
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid expanded record at line {line_number}") from None
                    fields = (
                        "document_sha256", "text", "split", "source_id", "source_dataset",
                        "source_revision", "source_license", "source_url",
                    )
                    if not isinstance(record, dict) or any(
                        not isinstance(record.get(key), str) or not record[key].strip() for key in fields
                    ):
                        raise ValueError(f"Missing expanded source provenance at line {line_number}")
                    key = record["document_sha256"]
                    if key in seen or text_hash(record["text"]) != key or record["split"] not in SPLITS:
                        raise ValueError(f"Invalid/duplicate expanded document at line {line_number}")
                    seen.add(key)
                    if key in existing:
                        if record != existing[key]:
                            raise ValueError("Existing candidate content, metadata or split assignment changed")
                        preserved.add(key)
                        continue
                    origin = (record["source_dataset"], record["source_revision"], record["source_license"])
                    identity = (record["source_dataset"], record["source_revision"], record["source_id"])
                    if origin not in allowed_origins or (identity in identities and identities[identity] != key):
                        raise ValueError("Expanded source origin or source-ID content conflicts with this run")
                    if tokenizer is not None and (
                        tokenizer.count(training_prompt(record["text"])) + self.config["summary_token_reserve"]
                        > self.config["context_limit"]
                    ):
                        raise ValueError(f"Expanded document exceeds the frozen token allowance at line {line_number}")
                    identities[identity] = key
                    self.db.execute(
                        "INSERT INTO documents (id,split,record) VALUES (?,?,?)",
                        (key, record["split"], json.dumps(record)),
                    )
                    added += 1
            if preserved != set(existing) or manifest.get("preserved_count") != len(existing):
                raise ValueError("Expanded preparation did not preserve every existing candidate")
            if sha256_file(path) != checksum:
                raise ValueError("Expanded input changed during import")
            extension = {
                "previous_documents_sha256": current_hash, "documents_sha256": checksum,
                "preserved_count": len(preserved), "added_count": added, "manifest": manifest,
            }
            self.db.execute(
                "INSERT OR REPLACE INTO metadata (key,value) VALUES ('candidate_extension',?)",
                (json.dumps(extension),),
            )
        atomic_json(self.run_dir / "candidate-extension.json", extension)
        return {"preserved": len(preserved), "added": added, "total": len(seen), "documents_sha256": checksum}

    def records(self, *, status: str | None = None, usable_only: bool = False):
        sql = "SELECT d.* FROM documents d"
        args = ()
        predicates = []
        if status is not None:
            predicates.append("d.status=?")
            args = (status,)
        if usable_only:
            predicates.extend((
                "d.status='accepted'",
                "NOT EXISTS (SELECT 1 FROM quality_holds h WHERE h.document_id=d.id)",
            ))
        if predicates:
            sql += " WHERE " + " AND ".join(predicates)
        for row in self.db.execute(sql + " ORDER BY d.id", args):
            value = dict(row)
            value["record"] = json.loads(value["record"])
            value["result"] = json.loads(value["result"]) if value["result"] else None
            yield value
