import json
from decimal import Decimal
from unittest.mock import patch

from summary_dataset_builder.files import load_json, sha256_file, text_hash
from summary_dataset_builder.generation import generate
from summary_dataset_builder.collection import collect
from summary_dataset_builder.cli import main
from summary_dataset_builder.export import export_dataset
from summary_dataset_builder.review import approve_review, review_sample
from summary_dataset_builder.store import RunStore
from tests.helpers import completion, FakeClient, RunCase


class CollectionPolicyTests(RunCase):
    def authorize(self, store, **overrides):
        arguments = {
            "budget_usd": Decimal("5"), "target_train": 2,
            "target_validation": 1, "target_test": 1, "acknowledge": True,
        }
        arguments.update(overrides)
        return store.authorize_collection(**arguments)

    def test_explicit_policy_extends_budget_but_does_not_reset_charges_or_approve_release(self):
        self.initialize()
        store = self.store()
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        original = load_json(self.run / "config.json")
        self.authorize(store)
        self.assertEqual(store.accounting()["known_cost_usd"], "0.0001")
        self.assertEqual(store.accounting()["budget_usd"], "5")
        self.assertEqual(load_json(self.run / "config.json"), original)
        with RunStore(self.run) as reopened:
            self.assertEqual(reopened.accounting()["budget_usd"], "5")
            self.assertEqual(reopened.accounting()["attempts"], 1)
            self.assertEqual(reopened.status()["approvals"], [])
        with self.assertRaisesRegex(ValueError, "explicit budget"):
            self.authorize(store, acknowledge=False)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            self.authorize(store, usable_threshold=Decimal(".5"))

    def test_quality_holds_preserve_original_decisions_and_require_replacement_samples(self):
        self.initialize()
        store = self.store()
        generate(store, approve_paid=True, max_requests=4, client=FakeClient())
        before = list(store.records())
        store.freeze_assessment("Assess.", {"type": "object"}, "test")
        source = store.next_assessment()
        attempt = store.reserve_assessment(source["id"], Decimal(".01"))
        store.finish_assessment(
            attempt, status="completed", cost=Decimal(".001"),
            result={"verdict": "concern", "disposition_correct": False},
        )
        self.authorize(store)
        self.assertEqual(list(store.records()), before)
        self.assertFalse(store.status()["generation_targets_met"])
        self.assertEqual(sum(store.usable_counts().values()), 3)
        self.assertEqual(len(list(store.records(usable_only=True))), 3)
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        self.assertTrue(store.status()["generation_targets_met"])
        self.assertEqual(len(list(store.records(usable_only=True))), 4)
        self.assertEqual(len(list(store.records(status="accepted"))), 5)

    def populate_stage(self, *, valid, usable):
        self.write_sources(count_per_split=60)
        self.initialize(target_train=60, target_validation=60, target_test=60)
        store = self.store()
        self.authorize(store, target_train=60, target_validation=60, target_test=60)
        for index in range(100):
            document = store.next_document()
            attempt = store.reserve(document["document_sha256"], Decimal(".001"))
            accepted = index < usable
            content = (
                json.dumps({
                    "accepted": accepted, "summary": "Supported summary." if accepted else "",
                    "rejection_reason": "" if accepted else "Unsuitable source.",
                }) if index < valid else "invalid JSON"
            )
            store.finish(
                attempt, status="accepted" if accepted else "rejected" if index < valid else "error",
                cost=Decimal("0"), response={"content": content, "finish_reason": "stop"},
                result={"accepted": accepted, "summary": "Supported summary." if accepted else "", "flags": []},
            )
        return store

    def test_exact_98_percent_structure_and_75_percent_usable_pass(self):
        store = self.populate_stage(valid=98, usable=75)
        with self.assertRaisesRegex(ValueError, "gate at 100"):
            store.reserve(store.next_document()["document_sha256"], Decimal(".001"))
        gate = store.evaluate_collection_gate()
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["structural_fraction"], "0.98")
        self.assertEqual(gate["usable_fraction"], "0.75")
        self.assertEqual(store.collection_status()["next_checkpoint"], 1000)
        self.assertIsNone(store.collection_status()["required_checkpoint"])
        self.assertEqual(store.status()["approvals"], [])
        store.reserve(store.next_document()["document_sha256"], Decimal(".001"))

    def test_structure_below_threshold_latches_failure(self):
        store = self.populate_stage(valid=97, usable=75)
        self.assertFalse(store.evaluate_collection_gate()["passed"])
        with self.assertRaisesRegex(ValueError, "gate failed"):
            store.reserve(store.next_document()["document_sha256"], Decimal(".001"))
        with self.assertRaisesRegex(ValueError, "already failed"):
            store.evaluate_collection_gate()

    def test_usable_below_threshold_latches_failure(self):
        store = self.populate_stage(valid=100, usable=74)
        gate = store.evaluate_collection_gate()
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["usable_fraction"], "0.74")

    def test_explicit_threshold_amendment_preserves_failed_gate_and_other_limits(self):
        store = self.populate_stage(valid=100, usable=74)
        failed = store.evaluate_collection_gate()
        original_policy = dict(store.collection_policy)
        original_records = list(store.records())
        original_accounting = store.accounting()
        base_config = load_json(self.run / "config.json")
        status = store.amend_usable_threshold(
            Decimal(".70"), reason="User explicitly approved 70% usable yield", acknowledge=True,
        )
        self.assertFalse(status["failed_gate"])
        self.assertEqual(status["required_checkpoint"], 100)
        self.assertEqual(store.collection_policy, {**original_policy, "usable_threshold": "0.70"})
        history = store.collection_policy_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["previous_policy"], original_policy)
        self.assertEqual(history[0]["superseded_stages"][0]["result"], failed)
        passed = store.evaluate_collection_gate()
        self.assertTrue(passed["passed"])
        for key in ("window_start", "attempted", "structurally_valid", "usable", "structural_fraction", "usable_fraction"):
            self.assertEqual(passed[key], failed[key])
        self.assertEqual(store.accounting(), original_accounting)
        self.assertEqual(list(store.records()), original_records)
        self.assertEqual(load_json(self.run / "config.json"), base_config)
        self.assertTrue((self.run / "collection-policy-amendment-1.json").is_file())
        with RunStore(self.run) as reopened:
            self.assertEqual(reopened.collection_policy["usable_threshold"], "0.70")
            self.assertEqual(reopened.collection_policy["structural_threshold"], "0.98")
            self.assertEqual(reopened.collection_policy["budget_usd"], "5")
            self.assertEqual(reopened.status()["approvals"], [])
            reopened.amend_usable_threshold(Decimal(".70"), reason="Same approved value", acknowledge=True)
            self.assertEqual(len(reopened.collection_policy_history()), 1)

    def test_lower_usable_threshold_does_not_override_structural_failure(self):
        store = self.populate_stage(valid=97, usable=74)
        self.assertFalse(store.evaluate_collection_gate()["passed"])
        store.amend_usable_threshold(Decimal(".70"), reason="Explicit user change", acknowledge=True)
        self.assertFalse(store.evaluate_collection_gate()["passed"])
        self.assertTrue(store.collection_status()["failed_gate"])

    def test_amendment_requires_explicit_approval_and_stopped_worker(self):
        self.initialize()
        store = self.store()
        self.authorize(store)
        with self.assertRaisesRegex(ValueError, "explicit approval"):
            store.amend_usable_threshold(Decimal(".70"), reason="No approval", acknowledge=False)
        with self.assertRaisesRegex(ValueError, "only lowers"):
            store.amend_usable_threshold(Decimal(".80"), reason="Unsupported tightening", acknowledge=True)
        lock = self.run / ".collection.lock"
        lock.write_text("active worker", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stop the collector"):
            store.amend_usable_threshold(Decimal(".70"), reason="Explicit user change", acknowledge=True)
        self.assertEqual(store.collection_policy_history(), [])

    def test_cli_amendment_allows_runner_to_recheck_and_resume(self):
        store = self.populate_stage(valid=100, usable=74)
        store.evaluate_collection_gate()
        with patch("builtins.print"):
            code = main([
                "amend-collection-policy", "--run", str(self.run), "--usable-threshold", "0.70",
                "--reason", "User approved lower usable yield", "--approve-policy-change",
            ])
        self.assertEqual(code, 0)
        with RunStore(self.run) as reopened:
            result = collect(reopened, approve_paid=True, max_requests=1, client=FakeClient())
            self.assertEqual(result["stop_reason"], "request_limit")
            self.assertEqual(reopened.accounting()["generation_attempts"], 101)
            self.assertTrue(reopened._collection_stages()[0]["passed"])
            self.assertEqual(reopened.status()["approvals"], [])

    def populate_transport_stage(self, *, legacy=False, all_transport=False):
        self.write_sources(count_per_split=60)
        self.initialize(target_train=60, target_validation=60, target_test=60)
        store = self.store()
        self.authorize(
            store, target_train=60, target_validation=60, target_test=60, usable_threshold=Decimal(".70")
        )
        if legacy:
            policy = dict(store.collection_policy)
            policy.pop("structural_denominator")
            with store.db:
                store.db.execute("UPDATE metadata SET value=? WHERE key='collection_policy'", (json.dumps(policy),))
            store._apply_collection_policy()
        for index in range(100):
            attempt = store.reserve(store.next_document()["document_sha256"], Decimal(".001"))
            if index < 6 or all_transport:
                store.finish(attempt, status="error", cost=Decimal(0), reason="http_429")
            else:
                accepted = index < 77
                result = {
                    "accepted": accepted, "summary": "Supported summary." if accepted else "",
                    "rejection_reason": "" if accepted else "Unsuitable source.",
                }
                store.finish(
                    attempt, status="accepted" if accepted else "rejected", cost=Decimal(".0001"),
                    result={**result, "flags": []},
                    response={"content": json.dumps(result), "finish_reason": "stop"},
                )
        return store

    def test_rate_limits_are_not_invalid_model_json_but_still_reduce_usable_yield(self):
        store = self.populate_transport_stage()
        gate = store.evaluate_collection_gate()
        self.assertEqual(gate["attempted"], 100)
        self.assertEqual(gate["model_responses"], 94)
        self.assertEqual(gate["transport_failures"], 6)
        self.assertEqual(gate["structural_fraction"], "1")
        self.assertEqual(gate["usable_fraction"], "0.71")
        self.assertTrue(gate["passed"])
        self.assertEqual(store.accounting()["generation_attempts"], 100)

    def test_metric_correction_archives_old_failure_and_preserves_thresholds_and_counts(self):
        store = self.populate_transport_stage(legacy=True)
        failed = store.evaluate_collection_gate()
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["structural_fraction"], "0.94")
        policy = dict(store.collection_policy)
        accounting = store.accounting()
        records = list(store.records())
        with self.assertRaisesRegex(ValueError, "explicit approval"):
            store.correct_structural_metric(acknowledge=False)
        store.correct_structural_metric(acknowledge=True)
        gate = store.evaluate_collection_gate()
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["structural_fraction"], "1")
        self.assertEqual(gate["usable_fraction"], failed["usable_fraction"])
        self.assertEqual(gate["window_start"], failed["window_start"])
        self.assertEqual(gate["attempted"], failed["attempted"])
        self.assertEqual(store.collection_policy, {**policy, "structural_denominator": "model_responses"})
        self.assertEqual(store.accounting(), accounting)
        self.assertEqual(list(store.records()), records)
        self.assertEqual(store.collection_policy_history()[0]["superseded_stages"][0]["result"], failed)
        self.assertEqual(store.status()["approvals"], [])
        store.correct_structural_metric(acknowledge=True)
        self.assertEqual(len(store.collection_policy_history()), 1)

    def test_actual_malformed_completions_remain_blocked_after_metric_correction(self):
        store = self.populate_stage(valid=97, usable=75)
        self.assertFalse(store.evaluate_collection_gate()["passed"])
        store.correct_structural_metric(acknowledge=True)
        self.assertTrue(store.collection_status()["failed_gate"])
        self.assertEqual(store.collection_policy_history(), [])

    def test_no_model_responses_cannot_pass_structural_gate(self):
        store = self.populate_transport_stage(all_transport=True)
        gate = store.evaluate_collection_gate()
        self.assertEqual(gate["model_responses"], 0)
        self.assertEqual(gate["structural_fraction"], "0")
        self.assertFalse(gate["passed"])

    def expanded_candidates(self):
        originals = [
            json.loads(line) for line in (self.prepared / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        additional = dict(originals[0])
        additional["source_id"] = "new-source"
        additional["text"] = "A completely separate source discusses astronomy and observations."
        additional["document_sha256"] = text_hash(additional["text"])
        output = self.root / "expanded"
        output.mkdir()
        (output / "documents.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in originals + [additional]), encoding="utf-8"
        )
        (output / "manifest.json").write_text(json.dumps({
            "format_version": 1, "documents_sha256": sha256_file(output / "documents.jsonl"),
            "preserved_documents_sha256": sha256_file(self.prepared / "documents.jsonl"),
            "preserved_count": len(originals),
        }), encoding="utf-8")
        return output

    def test_candidate_extension_preserves_existing_splits_responses_and_costs(self):
        self.initialize()
        store = self.store()
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        before = {row["id"]: row for row in store.records()}
        accounting = store.accounting()
        output = self.expanded_candidates()
        result = store.extend_candidates(output, acknowledge=True)
        self.assertEqual(result["preserved"], 12)
        self.assertEqual(result["added"], 1)
        after = {row["id"]: row for row in store.records()}
        for key, value in before.items():
            self.assertEqual(after[key], value)
        self.assertEqual(store.accounting(), accounting)

    def test_mutated_existing_split_rolls_back_entire_extension(self):
        self.initialize()
        store = self.store()
        output = self.expanded_candidates()
        path = output / "documents.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["split"] = "test"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = load_json(output / "manifest.json")
        manifest["documents_sha256"] = sha256_file(path)
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "split assignment changed"):
            store.extend_candidates(output, acknowledge=True)
        self.assertEqual(len(list(store.records())), 12)

    def test_export_requires_human_approval_and_omits_quality_holds(self):
        self.initialize(model_neutral=True, tokenizer_path=None)
        store = self.store()
        generate(store, approve_paid=True, max_requests=4, client=FakeClient())
        store.freeze_assessment("Assess.", {"type": "object"}, "test")
        record = store.next_assessment()
        attempt = store.reserve_assessment(record["id"], Decimal(".01"))
        store.finish_assessment(
            attempt, status="completed", cost=Decimal(".001"),
            result={"verdict": "concern", "disposition_correct": False},
        )
        self.authorize(store)
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        self.assertTrue(store.evaluate_collection_gate(final=True)["passed"])
        output = self.root / "export"
        with self.assertRaisesRegex(ValueError, "human-reviewed"):
            export_dataset(store, output)
        review = self.root / "review.json"
        review_sample(store, review)
        data = load_json(review)
        held_item = next(item for item in data["items"] if item["document_id"] == record["id"])
        self.assertIn("quality_hold", held_item)
        for item in data["items"]:
            item["verdict"] = "pass"
            item["notes"] = "Synthetic test only: source/disposition and quarantine checked."
        review.write_text(json.dumps(data), encoding="utf-8")
        approve_review(store, review, stage="export", acknowledge=True)
        result = export_dataset(store, output)
        self.assertEqual(result["counts"], {"train": 2, "validation": 1, "test": 1})
        exported_ids = {
            json.loads(line)["id"] for split in ("train", "validation", "test")
            for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        }
        self.assertNotIn(record["id"], exported_ids)
        self.assertEqual(len(exported_ids), 4)
        self.assertTrue((output / "quality-holds.jsonl").is_file())
