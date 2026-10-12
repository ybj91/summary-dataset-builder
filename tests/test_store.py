import json
from decimal import Decimal
from unittest.mock import patch

from summary_dataset_builder.files import load_json
from summary_dataset_builder.store import money, RunStore
from tests.helpers import RunCase


class StoreTests(RunCase):
    def test_initializes_and_preserves_provenance(self):
        self.initialize()
        store = self.store()
        row = next(store.records())
        self.assertEqual(row["record"]["source_revision"], "a" * 40)
        self.assertEqual(store.status()["accounting"]["attempts"], 0)
        self.assertFalse(store.status()["generation_targets_met"])

    def test_budget_reserved_before_request_and_not_reset_by_restart(self):
        self.initialize(budget_usd="0.02")
        store = self.store()
        document = store.next_document()
        attempt = store.reserve(document["document_sha256"], Decimal("0.015"))
        with RunStore(self.run) as reopened:
            self.assertEqual(reopened.accounting()["committed_usd"], "0.015")
            self.assertEqual(reopened.status()["inflight_attempts"], [attempt])
            with self.assertRaisesRegex(ValueError, "in-flight"):
                reopened.reserve(document["document_sha256"], Decimal("0.001"))
        store.finish(attempt, status="accepted", cost=Decimal("0.015"))
        with self.assertRaisesRegex(ValueError, "Budget"):
            store.reserve(store.next_document()["document_sha256"], Decimal("0.01"))
        self.assertEqual(store.accounting()["attempts"], 1)

    def test_unknown_cost_holds_entire_reservation_until_reconciled(self):
        self.initialize()
        store = self.store()
        attempt = store.reserve(store.next_document()["document_sha256"], Decimal("0.1"))
        store.finish(attempt, status="accepted", cost=None, result={"summary": "Saved", "flags": []})
        self.assertEqual(store.accounting()["uncertain_reserved_usd"], "0.1")
        self.assertEqual(next(store.records(status="uncertain"))["result"]["summary"], "Saved")
        store.reconcile(attempt, Decimal("0.003"))
        self.assertEqual(store.accounting()["uncertain_attempts"], 0)
        self.assertEqual(len(list(store.records(status="accepted"))), 1)

    def test_interrupted_request_is_never_automatically_retried(self):
        self.initialize()
        store = self.store()
        attempt = store.reserve(store.next_document()["document_sha256"], Decimal("0.1"))
        store.recover(attempt)
        self.assertEqual(len(list(store.records(status="uncertain"))), 1)
        store.reconcile(attempt, Decimal("0.005"), retry=True)
        self.assertEqual(store.accounting()["committed_usd"], "0.005")
        self.assertEqual(store.accounting()["attempts"], 1)
        self.assertEqual(len(list(store.records(status="pending"))), 12)

    def test_frozen_config_and_tokenizer_cannot_change_on_resume(self):
        self.initialize()
        path = self.run / "config.json"
        config = load_json(path)
        config["budget_usd"] = "1000"
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            RunStore(self.run)

    def test_prepared_input_hash_must_match_manifest(self):
        path = self.prepared / "documents.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.initialize()
        self.assertFalse(self.run.exists())

    def test_provider_reservation_overrun_blocks_future_requests(self):
        self.initialize()
        store = self.store()
        attempt = store.reserve(store.next_document()["document_sha256"], Decimal("0.001"))
        store.finish(attempt, status="error", cost=Decimal("0.002"))
        with self.assertRaisesRegex(ValueError, "exceeded its reservation"):
            store.reserve(store.next_document()["document_sha256"], Decimal("0.001"))

    def test_preparation_tokenizer_protocol_cannot_silently_change(self):
        path = self.prepared / "manifest.json"
        manifest = load_json(path)
        manifest["arguments"] = {
            "tokenizer_sha256": "0" * 64, "context_limit": 2048, "summary_token_reserve": 256
        }
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tokenizer/context/reserve"):
            self.initialize()

    def test_training_template_is_frozen_across_resume(self):
        self.initialize()
        with patch("summary_dataset_builder.store.training_prompt", return_value="new template"):
            with self.assertRaisesRegex(ValueError, "Training prompt template changed"):
                self.store()

    def test_model_neutral_run_does_not_load_a_tokenizer_or_freeze_a_template(self):
        with patch("summary_dataset_builder.store.LocalTokenizer", side_effect=AssertionError("No tokenizer")):
            self.initialize(tokenizer_path=None, model_neutral=True)
        self.assertFalse((self.run / "tokenizer.json").exists())
        with patch("summary_dataset_builder.store.training_prompt", return_value="different downstream template"):
            store = self.store()
        self.assertEqual(store.status()["dataset_mode"], "model-neutral")
        self.assertIsNone(store.config["tokenizer_sha256"])
        self.assertIsNone(store.config["context_limit"])

    def test_mode_is_explicit_and_mutually_exclusive(self):
        for kwargs in ({"tokenizer_path": None}, {"model_neutral": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Choose either"):
                self.initialize(**kwargs)

    def test_no_partial_run_when_source_length_or_counts_fail(self):
        for override in ({"context_limit": 260, "summary_token_reserve": 255}, {"target_train": 100}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.initialize(**override)
            self.assertFalse(self.run.exists())

    def test_review_gate_at_one_hundred_requests(self):
        self.write_sources(count_per_split=40)
        self.initialize(target_train=40, target_validation=40, target_test=40)
        store = self.store()
        for _ in range(100):
            attempt = store.reserve(store.next_document()["document_sha256"], Decimal("0.001"))
            store.finish(attempt, status="accepted", cost=Decimal("0"))
        with self.assertRaisesRegex(ValueError, "Review gate"):
            store.reserve(store.next_document()["document_sha256"], Decimal("0.001"))
        self.assertEqual(store.accounting()["attempts"], 100)

    def test_money_rejects_negative_nan_and_infinity(self):
        for value in ("NaN", "Infinity", "-0.1", True, "key-looking-invalid-value"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                money(value)
        self.assertEqual(money("0"), Decimal(0))

    def setup_assessment(self, **kwargs):
        self.initialize(**kwargs)
        store = self.store()
        document = store.next_document()
        attempt = store.reserve(document["document_sha256"], Decimal("0.01"))
        store.finish(
            attempt, status="accepted", cost=Decimal("0.005"),
            result={"accepted": True, "summary": "An unchanged summary", "rejection_reason": "", "flags": []},
            response={"content": "teacher-response", "request_id": "original"},
        )
        store.freeze_assessment("Assess source support.", {"type": "object"}, "assessor-v1.txt")
        return store, document["document_sha256"]

    def test_assessment_costs_share_budget_without_mutating_generation(self):
        store, document_id = self.setup_assessment(budget_usd="0.02")
        before = list(store.records())
        attempt = store.reserve_assessment(document_id, Decimal("0.014"))
        self.assertEqual(store.accounting()["committed_usd"], "0.019")
        with self.assertRaisesRegex(ValueError, "in-flight"):
            store.reserve(store.next_document()["document_sha256"], Decimal("0.001"))
        store.finish_assessment(
            attempt, status="completed", cost=Decimal("0.01"),
            result={"verdict": "concern"}, response={"content": "assessor-response"},
        )
        self.assertEqual(store.accounting()["known_cost_usd"], "0.015")
        self.assertEqual(store.accounting()["generation_attempts"], 1)
        self.assertEqual(store.accounting()["assessment_attempts"], 1)
        self.assertEqual(list(store.records()), before)
        self.assertIsNone(store.next_assessment())
        self.assertEqual(store.assessment_status()["verdicts"], {"concern": 1})
        with self.assertRaisesRegex(ValueError, "Budget"):
            store.reserve(store.next_document()["document_sha256"], Decimal("0.006"))

    def test_uncertain_assessment_recovery_keeps_original_accepted_record(self):
        store, document_id = self.setup_assessment()
        before = list(store.records())
        attempt = store.reserve_assessment(document_id, Decimal("0.01"))
        store.recover(attempt)
        self.assertEqual(store.assessment_status()["counts"], {"uncertain": 1})
        self.assertEqual(list(store.records()), before)
        store.reconcile(attempt, Decimal("0.003"), retry=True)
        self.assertEqual(store.assessment_status()["counts"], {"pending": 1})
        self.assertEqual(store.accounting()["uncertain_attempts"], 0)
        self.assertEqual(list(store.records()), before)
        retry = store.reserve_assessment(document_id, Decimal("0.01"))
        store.finish_assessment(retry, status="completed", cost=None, result={"verdict": "clear"})
        with self.assertRaisesRegex(ValueError, "response already exists"):
            store.reconcile(retry, Decimal("0.002"), retry=True)
        store.reconcile(retry, Decimal("0.002"))
        self.assertEqual(store.assessment_status()["verdicts"], {"clear": 1})
        self.assertEqual(list(store.records()), before)

    def test_assessment_protocol_snapshot_is_frozen_and_generations_are_distinct(self):
        store, document_id = self.setup_assessment()
        store.freeze_assessment("Assess source support.", {"type": "object"}, "assessor-v1.txt")
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            store.freeze_assessment("Different rubric", {"type": "object"}, "assessor-v2.txt")
        another = store.reserve(store.next_document()["document_sha256"], Decimal("0.01"))
        store.finish(another, status="rejected", cost=Decimal("0"), reason="Incomplete source")
        self.assertEqual(store.next_assessment()["id"], document_id)
        self.assertEqual(store.next_assessment()["response"]["request_id"], "original")
        self.assertEqual(store.assessment_status()["counts"], {"pending": 1})

    def test_authorized_request_limit_counts_both_generation_and_assessment(self):
        store, document_id = self.setup_assessment()
        store.authorize_request_limit(2, budget_usd=Decimal("1"), acknowledge=True)
        attempt = store.reserve_assessment(document_id, Decimal("0.01"))
        store.finish_assessment(attempt, status="error", cost=Decimal("0"))
        with self.assertRaisesRegex(ValueError, "total request limit"):
            store.reserve(store.next_document()["document_sha256"], Decimal("0.001"))
        self.assertEqual(store.status()["approvals"], [])
        with self.assertRaisesRegex(ValueError, "frozen total run budget"):
            store.authorize_request_limit(1000, budget_usd=Decimal("100"), acknowledge=True)
        with self.assertRaisesRegex(ValueError, "explicit paid"):
            store.authorize_request_limit(1000, budget_usd=Decimal("1"), acknowledge=False)

    def test_assessment_completion_cannot_be_written_as_generation(self):
        store, document_id = self.setup_assessment()
        attempt = store.reserve_assessment(document_id, Decimal("0.01"))
        with self.assertRaisesRegex(ValueError, "kind does not match"):
            store.finish(attempt, status="accepted", cost=Decimal("0"))
        self.assertEqual(store.status()["inflight_attempts"], [attempt])

    def test_existing_attempts_migrate_as_generation_without_changing_costs(self):
        store, _ = self.setup_assessment()
        before = store.accounting()
        with store.db:
            store.db.execute("ALTER TABLE attempts DROP COLUMN kind")
        store.db.close()
        with RunStore(self.run) as reopened:
            self.assertEqual(reopened.accounting(), before)
            self.assertEqual(reopened.db.execute("SELECT kind FROM attempts").fetchone()["kind"], "generation")
