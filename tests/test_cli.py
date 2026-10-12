from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import random
import unittest
from unittest.mock import patch

from summary_dataset_builder.cli import main
from summary_dataset_builder.files import load_json, sha256_file
from tests.helpers import FakeClient, RunCase


class CliTests(RunCase):
    def call_cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(list(map(str, args)))
        return status, output.getvalue(), errors.getvalue()

    def test_download_and_generate_require_explicit_approval(self):
        with patch("summary_dataset_builder.cli.download_fineweb") as download:
            code, _, error = self.call_cli(
                "download", "--output", self.root / "source.jsonl", "--revision", "a" * 40, "--max-records", 1
            )
            self.assertEqual(code, 2)
            self.assertIn("allow-download", error)
            download.assert_not_called()
        self.initialize()
        with patch("summary_dataset_builder.generation.OpenRouterClient") as client:
            code, _, error = self.call_cli("generate", "--run", self.run)
            self.assertEqual(code, 2)
            self.assertIn("terms-reviewed", error)
            client.assert_not_called()

    def test_status_and_plan_work_without_credentials(self):
        self.initialize()
        with patch.dict("os.environ", {}, clear=True):
            code, output, error = self.call_cli("plan", "--run", self.run)
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output)["pending_requests"], 12)

    def test_source_shortfall_is_nonzero_and_reported(self):
        with patch("summary_dataset_builder.cli.prepare_jsonl", return_value={"shortfall": 2}):
            code, output, error = self.call_cli(
                "prepare", "--input", self.root / "source.jsonl", "--output", self.root / "prepared2",
                "--dataset", "synthetic", "--revision", "a" * 40, "--license", "CC0-1.0"
            )
        self.assertEqual((code, error), (2, ""))
        self.assertEqual(json.loads(output)["shortfall"], 2)

    def test_neutral_init_and_generation_do_not_need_tokenizer(self):
        with patch("summary_dataset_builder.store.LocalTokenizer", side_effect=AssertionError("No tokenizer")):
            code, output, error = self.call_cli(
                "init", "--prepared", self.prepared, "--run", self.run,
                "--model", "openai/gpt-oss-120b", "--provider", "test-provider",
                "--budget-usd", 1, "--input-price-per-million", 1, "--output-price-per-million", 2,
                "--model-neutral", "--target-train", 2, "--target-validation", 1, "--target-test", 1,
            )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output)["dataset_mode"], "model-neutral")
        with patch("summary_dataset_builder.generation.OpenRouterClient", return_value=FakeClient()):
            code, output, error = self.call_cli(
                "generate", "--run", self.run, "--approve-paid", "--terms-reviewed"
            )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["generation_targets_met"])

    def test_missing_streaming_dependency_is_a_clear_cli_error(self):
        with patch("summary_dataset_builder.cli.download_fineweb", side_effect=RuntimeError("Install datasets")):
            code, _, error = self.call_cli(
                "download", "--output", self.root / "source.jsonl", "--revision", "a" * 40,
                "--max-records", 1, "--allow-download"
            )
        self.assertEqual(code, 2)
        self.assertEqual(error, "Error: Install datasets\n")

    def test_assessment_requires_explicit_spending_scope_and_terms(self):
        self.initialize()
        with patch("summary_dataset_builder.store.RunStore.authorize_request_limit") as authorize:
            code, _, error = self.call_cli(
                "assess", "--run", self.run, "--max-total-requests", 1000, "--budget-usd", 1
            )
        self.assertEqual(code, 2)
        self.assertIn("--approve-paid", error)
        authorize.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "Optional tokenizer extra not installed")
    def test_offline_cli_prepare_generate_review_export(self):
        source = self.root / "source.jsonl"
        randomizer = random.Random(1234)
        vocabulary = ["word" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(500)]
        with source.open("w", encoding="utf-8") as stream:
            for index in range(16):
                text = " ".join(randomizer.sample(vocabulary, 60)) + "."
                stream.write(json.dumps({"id": str(index), "url": f"https://sample{index}.example/page", "text": text}) + "\n")
        prepared = self.root / "cli-prepared"
        code, _, error = self.call_cli(
            "prepare", "--input", source, "--output", prepared, "--dataset", "synthetic",
            "--revision", "sha256:" + sha256_file(source), "--license", "CC0-1.0",
            "--min-words", 20, "--max-words", 100, "--train-count", 4,
            "--validation-count", 2, "--test-count", 2, "--tokenizer", self.tokenizer,
        )
        self.assertEqual((code, error), (0, ""))
        code, _, error = self.call_cli(
            "init", "--prepared", prepared, "--run", self.run, "--model", "openai/gpt-oss-120b",
            "--provider", "test-provider", "--budget-usd", 1, "--input-price-per-million", 1,
            "--output-price-per-million", 2, "--tokenizer", self.tokenizer,
            "--target-train", 2, "--target-validation", 1, "--target-test", 1,
        )
        self.assertEqual((code, error), (0, ""))
        with patch("summary_dataset_builder.generation.OpenRouterClient", return_value=FakeClient()):
            code, output, error = self.call_cli(
                "generate", "--run", self.run, "--max-requests", 20, "--approve-paid", "--terms-reviewed"
            )
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(json.loads(output)["generation_targets_met"])
        review = self.root / "cli-review.json"
        code, _, error = self.call_cli("review-sample", "--run", self.run, "--output", review)
        self.assertEqual((code, error), (0, ""))
        data = load_json(review)
        for item in data["items"]:
            item["verdict"] = "pass"
            item["notes"] = "Offline mock fixture, not a factual-quality evaluation."
        review.write_text(json.dumps(data), encoding="utf-8")
        code, _, error = self.call_cli(
            "approve", "--run", self.run, "--review", review, "--stage", "export", "--acknowledge-review"
        )
        self.assertEqual((code, error), (0, ""))
        code, output, error = self.call_cli("export", "--run", self.run, "--output", self.root / "dataset")
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output)["counts"]["train"], 2)
