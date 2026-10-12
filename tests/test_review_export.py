import json
from unittest.mock import patch

from summary_dataset_builder.export import export_dataset
from summary_dataset_builder.files import load_json
from summary_dataset_builder.generation import generate
from summary_dataset_builder.review import approve_review, review_sample
from tests.helpers import completion, FakeClient, FakeTokenizer, RunCase


class ReviewExportTests(RunCase):
    def complete_run(self):
        self.initialize()
        store = self.store()
        generate(store, approve_paid=True, max_requests=20, client=FakeClient())
        return store

    def completed_review(self, store):
        path = self.root / "review.json"
        review_sample(store, path, per_stratum=3)
        data = load_json(path)
        for item in data["items"]:
            item["verdict"] = "pass"
            item["notes"] = "Synthetic fixture reviewed for contract testing."
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_review_samples_accepted_rejected_flagged_and_heldout(self):
        self.initialize()
        store = self.store()
        responses = [
            completion(json.dumps({"accepted": False, "summary": "", "rejection_reason": "Incomplete"})),
            completion(json.dumps({"accepted": True, "summary": "It yielded 123456 tons.", "rejection_reason": ""})),
        ] + [completion() for _ in range(4)]
        generate(store, approve_paid=True, max_requests=6, client=FakeClient(responses))
        path = self.root / "review.json"
        review_sample(store, path)
        items = load_json(path)["items"]
        self.assertEqual({item["status"] for item in items}, {"accepted", "rejected", "flagged"})
        self.assertEqual({item["source"]["split"] for item in items}, {"train", "validation", "test"})

    def test_human_review_required_and_cannot_delete_hard_examples(self):
        store = self.complete_run()
        path = self.root / "review.json"
        review_sample(store, path)
        with self.assertRaisesRegex(ValueError, "All reviewed"):
            approve_review(store, path, stage="export", acknowledge=True)
        data = load_json(path)
        data["items"].pop()
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "added or removed"):
            approve_review(store, path, stage="export", acknowledge=True)
        with self.assertRaisesRegex(ValueError, "human-reviewed"):
            export_dataset(store, self.root / "export")

    def test_exports_exact_targets_summary_only_and_provenance(self):
        store = self.complete_run()
        review = self.completed_review(store)
        approve_review(store, review, stage="export", acknowledge=True)
        output = self.root / "export"
        with patch("summary_dataset_builder.export.LocalTokenizer", FakeTokenizer):
            manifest = export_dataset(store, output)
        self.assertEqual(manifest["counts"], {"train": 2, "validation": 1, "test": 1})
        pairs = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(set(pairs[0]), {"prompt", "completion"})
        self.assertFalse(pairs[0]["completion"].startswith("{"))
        provenance = json.loads((output / "train.manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(provenance["source_revision"], "a" * 40)
        self.assertEqual(provenance["attempts"][0]["response"]["request_id"], "test-request")
        self.assertEqual(len((output / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()), 12)
        with self.assertRaisesRegex(ValueError, "already exists"):
            export_dataset(store, output)

    def test_token_overflow_aborts_atomically_instead_of_dropping_heldout(self):
        store = self.complete_run()
        approve_review(store, self.completed_review(store), stage="export", acknowledge=True)

        class TooLong(FakeTokenizer):
            def count(self, text):
                return 2049

        output = self.root / "export"
        with patch("summary_dataset_builder.export.LocalTokenizer", TooLong):
            with self.assertRaisesRegex(ValueError, "no truncation"):
                export_dataset(store, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("export.*")), [])

    def test_stale_review_is_rejected(self):
        self.initialize()
        store = self.store()
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        path = self.completed_review(store)
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        with self.assertRaisesRegex(ValueError, "stale"):
            approve_review(store, path, stage="export", acknowledge=True)

    def test_failed_human_review_blocks_approval(self):
        store = self.complete_run()
        path = self.completed_review(store)
        data = load_json(path)
        data["items"][0]["verdict"] = "fail"
        data["items"][0]["notes"] = "Key fact omitted."
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "investigate failures"):
            approve_review(store, path, stage="export", acknowledge=True)

    def complete_neutral_run(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        store = self.store()
        generate(store, approve_paid=True, max_requests=20, client=FakeClient())
        approve_review(store, self.completed_review(store), stage="export", acknowledge=True)
        return store

    def test_neutral_export_preserves_plain_source_summary_and_ids(self):
        store = self.complete_neutral_run()
        output = self.root / "neutral"
        with patch("summary_dataset_builder.export.LocalTokenizer", side_effect=AssertionError("No tokenizer")):
            report = export_dataset(store, output)
        self.assertEqual(report["format"], "source-summary-jsonl")
        self.assertIsNone(report["tokenizer_sha256"])
        self.assertIsNone(report["context_limit"])
        self.assertIsNone(report["training_prompt_sha256"])
        pair = json.loads((output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(pair), {"id", "source", "summary"})
        by_id = {row["id"]: row for row in store.records(status="accepted")}
        self.assertEqual(pair["source"], by_id[pair["id"]]["record"]["text"])
        self.assertEqual(pair["summary"], by_id[pair["id"]]["result"]["summary"])
        provenance = json.loads((output / "train.manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertIsNone(provenance["training_tokens"])
        self.assertEqual(provenance["document_sha256"], pair["id"])

    def test_neutral_run_can_export_optional_token_checked_adapter_without_mutation(self):
        store = self.complete_neutral_run()
        original = dict(store.config)
        with patch("summary_dataset_builder.export.LocalTokenizer", FakeTokenizer):
            report = export_dataset(
                store, self.root / "adapter", tokenizer_path=self.tokenizer, context_limit=4096
            )
        self.assertEqual(report["format"], "prompt-completion-jsonl")
        self.assertEqual(report["context_limit"], 4096)
        self.assertIsNotNone(report["tokenizer_sha256"])
        self.assertEqual(store.config, original)
        canonical = export_dataset(store, self.root / "canonical")
        self.assertEqual(canonical["format"], "source-summary-jsonl")

    def test_neutral_export_still_requires_human_review(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        store = self.store()
        generate(store, approve_paid=True, max_requests=20, client=FakeClient())
        with self.assertRaisesRegex(ValueError, "human-reviewed"):
            export_dataset(store, self.root / "neutral")

    def test_neutral_export_adapter_overflow_does_not_remove_examples(self):
        store = self.complete_neutral_run()
        with patch("summary_dataset_builder.export.LocalTokenizer", FakeTokenizer):
            with self.assertRaisesRegex(ValueError, "no truncation"):
                export_dataset(store, self.root / "adapter", tokenizer_path=self.tokenizer, context_limit=1)
        self.assertFalse((self.root / "adapter").exists())
        self.assertTrue(store.status()["generation_targets_met"])

    def test_export_rejects_context_without_tokenizer_and_frozen_run_overrides(self):
        store = self.complete_neutral_run()
        with self.assertRaisesRegex(ValueError, "requires an export tokenizer"):
            export_dataset(store, self.root / "neutral", context_limit=4096)
        with patch("summary_dataset_builder.export.LocalTokenizer", FakeTokenizer):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                export_dataset(store, self.root / "adapter", tokenizer_path=self.tokenizer, context_limit=0)

    def test_assessor_requests_cannot_satisfy_generation_review_gate(self):
        store = self.complete_run()
        review = self.completed_review(store)
        accounting = store.accounting()
        accounting.update({"attempts": 100, "assessment_attempts": 96})
        with patch.object(store, "accounting", return_value=accounting):
            with self.assertRaisesRegex(ValueError, "100 generation attempts"):
                approve_review(store, review, stage="scale", acknowledge=True)
        self.assertEqual(store.status()["approvals"], [])
