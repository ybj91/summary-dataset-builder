import json
import os
from decimal import Decimal
import unittest
from unittest.mock import patch

from summary_dataset_builder.export import export_dataset
from summary_dataset_builder.collection import collect
from summary_dataset_builder.files import load_json
from summary_dataset_builder.generation import generate
from summary_dataset_builder.review import approve_review, review_sample
from tests.helpers import FakeClient, FakeTokenizer, RunCase


@unittest.skipUnless(os.environ.get("RUN_SCALE_TESTS") == "1", "Opt-in 10k offline durability test")
class ScaleTests(RunCase):
    def approve_fixture(self, store, stage):
        path = self.root / f"review-{stage}.json"
        review_sample(store, path)
        sample = load_json(path)
        for item in sample["items"]:
            item["verdict"] = "pass"
            item["notes"] = "Synthetic offline scale fixture; not a real human quality approval."
        path.write_text(json.dumps(sample), encoding="utf-8")
        approve_review(store, path, stage=stage, acknowledge=True)

    def test_ten_thousand_training_pairs_with_review_gates_and_resume(self):
        self.write_sources(count_per_split=10000)
        self.initialize(target_train=10000, target_validation=200, target_test=200, budget_usd="5")
        store = self.store()
        for size, stage in ((100, "scale"), (900, "full")):
            output = generate(store, approve_paid=True, max_requests=size, client=FakeClient())
            self.assertEqual(output["requests_this_invocation"], size)
            with self.assertRaisesRegex(ValueError, "Review gate"):
                generate(store, approve_paid=True, max_requests=1, client=FakeClient())
            self.approve_fixture(store, stage)
        store.db.close()
        store = self.store()
        output = generate(store, approve_paid=True, max_requests=15000, client=FakeClient())
        self.assertEqual(output["stop_reason"], "targets_met")
        self.assertEqual(output["counts"]["train"]["accepted"], 10000)
        self.assertEqual(output["accounting"]["attempts"], 10400)
        self.approve_fixture(store, "export")
        with patch("summary_dataset_builder.export.LocalTokenizer", FakeTokenizer):
            exported = export_dataset(store, self.root / "export")
        self.assertEqual(exported["counts"], {"train": 10000, "validation": 200, "test": 200})
        with (self.root / "export" / "train.jsonl").open(encoding="utf-8") as stream:
            self.assertEqual(sum(1 for _ in stream), 10000)

    def test_automatic_collection_reaches_exact_10k_without_fabricating_human_approval(self):
        self.write_sources(count_per_split=10000)
        self.initialize(model_neutral=True, tokenizer_path=None)
        store = self.store()
        store.authorize_collection(
            budget_usd=Decimal("5"), target_train=10000, target_validation=200, target_test=200,
            acknowledge=True,
        )
        result = collect(store, approve_paid=True, client=FakeClient())
        self.assertEqual(result["stop_reason"], "targets_met")
        status = store.collection_status()
        self.assertEqual(status["usable_counts"], {"train": 10000, "validation": 200, "test": 200})
        self.assertEqual(status["accounting"]["generation_attempts"], 10400)
        self.assertEqual(status["accounting"]["known_cost_usd"], "1.0400")
        self.assertEqual(status["approvals"], [])
        self.assertFalse(status["failed_gate"])
        self.assertEqual(
            [stage["generation_attempts"] for stage in status["stages"]],
            [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10400],
        )
        self.assertTrue((self.run / "collection-progress.json").is_file())
        self.assertFalse((self.run / ".collection.lock").exists())
        with self.assertRaisesRegex(ValueError, "human-reviewed"):
            export_dataset(store, self.root / "not-authorized")
