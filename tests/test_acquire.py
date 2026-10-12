import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from summary_dataset_builder import acquire
from summary_dataset_builder.acquire import download_fineweb
from summary_dataset_builder.prepare import prepare_jsonl


REVISION = "b" * 40


def make_record(index):
    return {
        "id": f"record-{index}",
        "url": f"https://example.org/article/{index}",
        "text": f"Example article number {index}.",
        "quality": {"score": 0.9},
    }


class CountingStream:
    def __init__(self, total):
        self.total = total
        self.requested = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.requested += 1
        if self.requested > self.total:
            raise StopIteration
        return make_record(self.requested)


class AcquireTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".cache") / f"acquire-tests-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.output_path = self.root / "download.jsonl"
        self.metadata_path = self.root / "download.jsonl.metadata.json"
        self.rejections_path = self.root / "download.jsonl.rejections.jsonl"
        self.loader = Mock(return_value=[make_record(index) for index in range(3)])
        self.fake_datasets = SimpleNamespace(load_dataset=self.loader)
        self.module_patch = patch.dict(sys.modules, {"datasets": self.fake_datasets})
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()
        shutil.rmtree(self.root)

    def download(self, **overrides):
        arguments = {"revision": REVISION, "max_records": 2}
        arguments.update(overrides)
        return download_fineweb(self.output_path, **arguments)

    def assert_no_output(self):
        self.assertFalse(self.output_path.exists())
        self.assertFalse(self.metadata_path.exists())
        self.assertFalse(self.rejections_path.exists())
        self.assertFalse(list(self.root.glob(".*.building-*")))

    def test_pinned_lazy_streaming_arguments_and_metadata(self):
        result = self.download()
        self.loader.assert_called_once_with(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
            revision=REVISION, streaming=True,
        )
        rows = [json.loads(line) for line in self.output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {**make_record(0), "source_split": "train"})
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["format_version"], 1)
        self.assertEqual(metadata["dataset"], "HuggingFaceFW/fineweb-edu")
        self.assertEqual(metadata["revision"], REVISION)
        self.assertEqual(metadata["subset"], "sample-10BT")
        self.assertEqual(metadata["split"], "train")
        self.assertEqual(metadata["limit"], 2)
        self.assertEqual(metadata["records_written"], 2)
        self.assertEqual(metadata["inspected_count"], 2)
        self.assertEqual(metadata["downloaded_count"], 2)
        self.assertEqual(metadata["rejected_count"], 0)
        self.assertEqual(metadata["rejected_counts"], {})
        self.assertEqual(metadata["shortfall"], 0)
        self.assertEqual(metadata["download_sha256"], hashlib.sha256(self.output_path.read_bytes()).hexdigest())
        self.assertEqual(metadata["rejections_sha256"], hashlib.sha256(self.rejections_path.read_bytes()).hexdigest())
        self.assertEqual(metadata["rejections_file"], self.rejections_path.name)
        self.assertEqual(result["rejections_path"], str(self.rejections_path))
        self.assertEqual(result["download_sha256"], metadata["download_sha256"])
        self.assertFalse(metadata["record_limit_is_byte_limit"])
        self.assertEqual(set(path.name for path in self.root.iterdir()), {
            "download.jsonl", "download.jsonl.metadata.json", "download.jsonl.rejections.jsonl",
        })

    def test_iterator_is_not_advanced_beyond_requested_bound(self):
        stream = CountingStream(total=100)
        self.loader.return_value = stream
        result = self.download(max_records=3)
        self.assertEqual(stream.requested, 3)
        self.assertEqual(result["records_written"], 3)
        self.assertTrue(result["record_limit_reached"])

    def test_exhausted_dataset_has_explicit_count(self):
        self.loader.return_value = [make_record(1)]
        result = self.download(max_records=5)
        self.assertEqual(result["records_written"], 1)
        self.assertFalse(result["record_limit_reached"])

    def test_empty_dataset_has_valid_empty_hash(self):
        self.loader.return_value = []
        result = self.download()
        self.assertEqual(result["records_written"], 0)
        self.assertEqual(result["download_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(result["rejections_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(result["shortfall"], 2)
        self.assertEqual(result["inspected_count"], 0)

    def test_custom_dataset_subset_and_source_split(self):
        self.loader.return_value = [{**make_record(1), "source_split": "untrusted"}]
        result = self.download(dataset="owner/source", subset="subset-a", split="validation")
        self.loader.assert_called_once_with(
            "owner/source", name="subset-a", split="validation", revision=REVISION, streaming=True,
        )
        row = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(row["source_split"], "validation")
        self.assertEqual(result["dataset"], "owner/source")

    def test_invalid_arguments_never_import_or_download(self):
        invalid = [
            {"revision": "main"}, {"revision": "sha256:" + "a" * 64},
            {"revision": "z" * 40}, {"revision": "a" * 39},
            {"max_records": 0}, {"max_records": -1}, {"max_records": True},
            {"max_records": 1.5}, {"dataset": ""}, {"subset": " "}, {"split": None},
        ]
        with patch.object(acquire.importlib, "import_module", side_effect=AssertionError("must not import")):
            for overrides in invalid:
                with self.subTest(overrides=overrides):
                    with self.assertRaises(ValueError):
                        self.download(**overrides)
                    self.assert_no_output()
        self.loader.assert_not_called()

    def test_importing_module_never_imports_datasets_or_downloads(self):
        original_import_module = importlib.import_module
        with patch.object(
            importlib, "import_module", side_effect=AssertionError("must not import datasets"),
        ):
            importlib.reload(acquire)
        self.assertIs(importlib.import_module, original_import_module)
        self.loader.assert_not_called()
        self.assert_no_output()

    def test_optional_dependency_failure_is_actionable(self):
        with patch.object(acquire.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "optional 'datasets'"):
                self.download()
        self.loader.assert_not_called()
        self.assert_no_output()

    def test_existing_artifacts_are_not_overwritten(self):
        for existing in (self.output_path, self.rejections_path, self.metadata_path):
            with self.subTest(existing=existing.name):
                existing.write_text("preserve", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    self.download()
                self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")
                self.loader.assert_not_called()
                existing.unlink()

    def test_malformed_row_has_location_and_atomic_cleanup(self):
        bad_rows = [
            {"id": "bad", "url": "https://example.org", "text": 1},
            {"id": 3, "url": "https://example.org", "text": "sensitive source"},
            {"id": "", "url": "https://example.org", "text": "sensitive source"},
            {"id": "missing-text", "url": "https://example.org"},
            {"id": "bad", "url": "https://example.org", "text": "sensitive source", "meta": object()},
            {"id": "bad", "url": "https://example.org", "text": "sensitive source", "score": float("nan")},
            {"id": "bad", "url": "https://example.org", "text": "\ud800"},
            ["not a record"],
        ]
        for bad in bad_rows:
            with self.subTest(row_type=type(bad).__name__):
                self.loader.return_value = [make_record(1), bad]
                with self.assertRaises(ValueError) as caught:
                    self.download()
                self.assertIn("dataset record 2:", str(caught.exception))
                self.assertNotIn("sensitive source", str(caught.exception))
                self.assert_no_output()

    def test_stream_initialization_failure_is_explicit_and_sanitized(self):
        self.loader.side_effect = OSError("a-sensitive-token")
        with self.assertRaisesRegex(RuntimeError, "initialization failed") as caught:
            self.download()
        self.assertNotIn("a-sensitive-token", str(caught.exception))
        self.assert_no_output()

    def test_interrupted_stream_does_not_publish_partial_data(self):
        def interrupted():
            yield make_record(1)
            raise OSError("a-sensitive-token")

        self.loader.return_value = interrupted()
        with self.assertRaisesRegex(RuntimeError, "before record 2") as caught:
            self.download()
        self.assertNotIn("a-sensitive-token", str(caught.exception))
        self.assert_no_output()

    def test_keyboard_interrupt_cleans_staged_data(self):
        def interrupted():
            yield make_record(1)
            raise KeyboardInterrupt

        self.loader.return_value = interrupted()
        with self.assertRaises(KeyboardInterrupt):
            self.download()
        self.assert_no_output()

    def test_metadata_publication_failure_rolls_back_data_and_rejections(self):
        real_publish = acquire._publish_exclusive

        def fail_metadata(staged, destination):
            if destination == self.metadata_path:
                raise OSError("simulated metadata failure")
            real_publish(staged, destination)

        with patch.object(acquire, "_publish_exclusive", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                self.download()
        self.assert_no_output()

    def test_racing_existing_metadata_is_preserved(self):
        real_publish = acquire._publish_exclusive

        def race_metadata(staged, destination):
            if destination == self.metadata_path:
                self.metadata_path.write_text("racing writer", encoding="utf-8")
            real_publish(staged, destination)

        with patch.object(acquire, "_publish_exclusive", side_effect=race_metadata):
            with self.assertRaises(FileExistsError):
                self.download()
        self.assertFalse(self.output_path.exists())
        self.assertFalse(self.rejections_path.exists())
        self.assertEqual(self.metadata_path.read_text(encoding="utf-8"), "racing writer")
        self.assertFalse(list(self.root.glob(".*.building-*")))

    def test_racing_existing_data_is_preserved(self):
        real_publish = acquire._publish_exclusive

        def race_data(staged, destination):
            if destination == self.output_path:
                self.output_path.write_text("racing writer", encoding="utf-8")
            real_publish(staged, destination)

        with patch.object(acquire, "_publish_exclusive", side_effect=race_data):
            with self.assertRaises(FileExistsError):
                self.download()
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "racing writer")
        self.assertFalse(self.metadata_path.exists())
        self.assertFalse(self.rejections_path.exists())

    def test_rejection_publication_failure_rolls_back_data(self):
        real_publish = acquire._publish_exclusive

        def fail_rejections(staged, destination):
            if destination == self.rejections_path:
                raise OSError("simulated rejection sidecar publication failure")
            real_publish(staged, destination)

        with patch.object(acquire, "_publish_exclusive", side_effect=fail_rejections):
            with self.assertRaises(OSError):
                self.download()
        self.assert_no_output()

    def test_racing_existing_rejections_are_preserved(self):
        real_publish = acquire._publish_exclusive

        def race_rejections(staged, destination):
            if destination == self.rejections_path:
                self.rejections_path.write_text("racing writer", encoding="utf-8")
            real_publish(staged, destination)

        with patch.object(acquire, "_publish_exclusive", side_effect=race_rejections):
            with self.assertRaises(FileExistsError):
                self.download()
        self.assertFalse(self.output_path.exists())
        self.assertFalse(self.metadata_path.exists())
        self.assertEqual(self.rejections_path.read_text(encoding="utf-8"), "racing writer")
        self.assertFalse(list(self.root.glob(".*.building-*")))

    def test_file_sync_failure_does_not_publish(self):
        with patch.object(acquire.os, "fsync", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                self.download()
        self.assert_no_output()

    def test_invalid_record_beyond_limit_is_never_validated(self):
        self.loader.return_value = [make_record(1), {"bad": "must not be inspected"}]
        result = self.download(max_records=1)
        self.assertEqual(result["records_written"], 1)

    def test_invalid_url_is_audited_and_counts_toward_inspection_bound(self):
        requested = []
        bad_url = "https://user:do-not-expose@example.org/private"
        bad_text = "private-document-text"
        records = [
            make_record(1), {**make_record(2), "url": bad_url, "text": bad_text},
            make_record(3),
        ]

        def bounded_source():
            for index, record in enumerate(records, 1):
                requested.append(index)
                yield record
            raise AssertionError("the source must not be advanced beyond three inspected records")

        self.loader.return_value = bounded_source()
        result = self.download(max_records=3)
        self.assertEqual(requested, [1, 2, 3])
        rows = [json.loads(line) for line in self.output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in rows], ["record-1", "record-3"])
        self.assertEqual(result["inspected_count"], 3)
        self.assertEqual(result["downloaded_count"], 2)
        self.assertEqual(result["records_written"], 2)
        self.assertEqual(result["rejected_count"], 1)
        self.assertEqual(result["rejected_counts"], {"invalid_source_url": 1})
        self.assertEqual(result["shortfall"], 1)
        self.assertTrue(result["record_limit_reached"])
        rejection = json.loads(self.rejections_path.read_text(encoding="utf-8"))
        self.assertEqual(rejection, {
            "source_id": "record-2", "source_id_is_hash": False,
            "ordinal": 2, "reason": "invalid_source_url",
        })
        self.assertEqual(result["download_sha256"], hashlib.sha256(self.output_path.read_bytes()).hexdigest())
        self.assertEqual(result["rejections_sha256"], hashlib.sha256(self.rejections_path.read_bytes()).hexdigest())
        published = "\n".join(path.read_text(encoding="utf-8") for path in self.root.iterdir())
        self.assertNotIn(bad_url, published)
        self.assertNotIn("do-not-expose", published)
        self.assertNotIn(bad_text, published)
        prepared = prepare_jsonl(
            self.output_path, self.root / "prepared", dataset="HuggingFaceFW/fineweb-edu",
            revision=REVISION, license_name="CC-BY-4.0", min_words=1,
            train_count=1, validation_count=1, test_count=1,
        )
        self.assertEqual(prepared["scanned_records"], 2)
        self.assertEqual(prepared["selected_records"], 2)
        self.assertEqual(prepared["shortfall"], 1)
        self.assertEqual(prepared["acquisition_metadata"]["inspected_count"], 3)
        self.assertEqual(prepared["acquisition_metadata"]["downloaded_count"], 2)
        documents = [
            json.loads(line)
            for line in (self.root / "prepared" / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual({row["source_url"] for row in documents}, {
            records[0]["url"], records[2]["url"],
        })

    def test_all_invalid_urls_produce_audited_empty_output_and_shortfall(self):
        urls = [
            "ftp://example.org", "https://example.org/has space", "https://example.org:wrong",
            "https://[invalid", "https://example.org/\x00", "https://@example.org",
            "https://example.org/\n", None, 9,
        ]
        rows = [{**make_record(index), "url": url} for index, url in enumerate(urls)]
        rows.append({"id": "missing-url", "text": "not a training source"})
        self.loader.return_value = rows
        result = self.download(max_records=len(rows))
        self.assertTrue(result["all_records_rejected"])
        self.assertEqual(result["downloaded_count"], 0)
        self.assertEqual(result["inspected_count"], len(rows))
        self.assertEqual(result["rejected_count"], len(rows))
        self.assertEqual(result["shortfall"], len(rows))
        self.assertEqual(self.output_path.read_bytes(), b"")
        self.assertEqual(result["download_sha256"], hashlib.sha256(b"").hexdigest())
        prepared = prepare_jsonl(
            self.output_path, self.root / "prepared", dataset="HuggingFaceFW/fineweb-edu",
            revision=REVISION, license_name="CC-BY-4.0", min_words=1,
            train_count=1, validation_count=1, test_count=1,
        )
        self.assertEqual(prepared["scanned_records"], 0)
        self.assertEqual(prepared["shortfall"], 3)

    def test_unsafe_rejection_ids_are_hashed_without_source_content(self):
        source_ids = [
            "https://user:private-password@example.org", "someone@example.org",
            "sk-abcdefghijklmnopqrstuv", "line\nbreak", "\ud800",
        ]
        self.loader.return_value = [
            {"id": source_id, "url": "not-a-valid-url", "text": "private raw text"}
            for source_id in source_ids
        ]
        self.download(max_records=len(source_ids))
        audited = [
            json.loads(line) for line in self.rejections_path.read_text(encoding="utf-8").splitlines()
        ]
        for index, (source_id, rejection) in enumerate(zip(source_ids, audited), 1):
            self.assertTrue(rejection["source_id_is_hash"])
            self.assertEqual(rejection["ordinal"], index)
            self.assertEqual(
                rejection["source_id"],
                "sha256:" + hashlib.sha256(source_id.encode("utf-8", errors="surrogatepass")).hexdigest(),
            )
        self.assertNotIn("private", self.rejections_path.read_text(encoding="utf-8"))
        self.assertNotIn("url", audited[0])
        self.assertNotIn("text", audited[0])

    def test_failure_location_uses_inspected_ordinal_after_url_rejection(self):
        self.loader.return_value = [
            {**make_record(1), "url": "invalid URL"},
            {"id": "missing-text", "url": "https://example.org"},
        ]
        with self.assertRaisesRegex(ValueError, "dataset record 2:"):
            self.download()
        self.assert_no_output()

    def test_stream_failure_after_url_rejection_uses_inspected_ordinal(self):
        def interrupted():
            yield {**make_record(1), "url": "invalid URL"}
            raise OSError("private-url-content")

        self.loader.return_value = interrupted()
        with self.assertRaisesRegex(RuntimeError, "before record 2") as caught:
            self.download()
        self.assertNotIn("private-url-content", str(caught.exception))
        self.assert_no_output()

    def test_rejection_write_failure_rolls_back_all_artifacts(self):
        self.loader.return_value = [make_record(1), {**make_record(2), "url": "invalid URL"}]
        real_write = acquire._write_json_line

        def fail_rejection_write(handle, value):
            if value.get("reason") == "invalid_source_url":
                raise OSError("simulated audit write failure")
            return real_write(handle, value)

        with patch.object(acquire, "_write_json_line", side_effect=fail_rejection_write):
            with self.assertRaises(OSError):
                self.download()
        self.assert_no_output()

    def test_download_preparation_checks_provenance_offline(self):
        self.download()
        arguments = {
            "dataset": "HuggingFaceFW/fineweb-edu", "revision": REVISION,
            "license_name": "CC-BY-4.0", "min_words": 1,
            "train_count": 1, "validation_count": 1, "test_count": 1,
        }
        result = prepare_jsonl(self.output_path, self.root / "prepared", **arguments)
        self.assertTrue(result["acquisition_metadata_verified"])
        self.assertEqual(result["acquisition_metadata"]["records_written"], 2)
        with self.assertRaisesRegex(ValueError, "revision does not match"):
            prepare_jsonl(
                self.output_path, self.root / "wrong-revision",
                **{**arguments, "revision": "c" * 40},
            )
        self.assertFalse((self.root / "wrong-revision").exists())
        self.loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
