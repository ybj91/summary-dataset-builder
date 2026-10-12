import hashlib
import json
from pathlib import Path
import random
import shutil
import unittest
from unittest.mock import patch
import uuid

from summary_dataset_builder import prepare
from summary_dataset_builder.prepare import prepare_jsonl
from summary_dataset_builder.tokenization import training_prompt


REVISION = "a" * 40


def make_record(index, *, words=30, domain=None):
    return {
        "id": f"source-{index}",
        "url": f"https://{domain or f'domain{index}.example'}/article/{index}",
        "text": " ".join(f"article{index}_word{word}" for word in range(words)),
    }


def make_anchor(index, *, split="train", words=30, domain=None):
    source = make_record(index, words=words, domain=domain)
    return {
        "source_id": source["id"], "source_url": source["url"],
        "source_dataset": "test/source", "source_revision": REVISION,
        "source_license": "CC-BY-4.0", "text": source["text"], "split": split,
        "source_split": "upstream-train",
        "document_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
        "normalized_sha256": hashlib.sha256(prepare._normalize(source["text"]).encode("utf-8")).hexdigest(),
        "review_metadata": {"approved": True, "notes": ["retain unchanged"], "score": 0.75},
    }


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".cache") / f"prepare-tests-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.input_path = self.root / "source.jsonl"
        self.output_dir = self.root / "prepared"

    def tearDown(self):
        shutil.rmtree(self.root)

    def write_records(self, records):
        with self.input_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_preserved(self, records, *, trailing_newline=True):
        path = self.root / "anchors.jsonl"
        raw = b"\r\n".join(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            for record in records
        )
        if raw and trailing_newline:
            raw += b"\r\n"
        path.write_bytes(raw)
        return path

    def write_acquisition_metadata(self, **overrides):
        metadata = {
            "format_version": 1, "dataset": "test/source", "revision": REVISION,
            "subset": "sample", "split": "train", "limit": 3, "records_written": 3,
            "download_sha256": hashlib.sha256(self.input_path.read_bytes()).hexdigest(),
        }
        metadata.update(overrides)
        path = self.input_path.with_name(self.input_path.name + ".metadata.json")
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def write_audited_acquisition_metadata(self, **overrides):
        rejections_path = self.input_path.with_name(self.input_path.name + ".rejections.jsonl")
        rejections_path.write_text(json.dumps({
            "source_id": "rejected-source", "source_id_is_hash": False,
            "ordinal": 2, "reason": "invalid_source_url",
        }) + "\n", encoding="utf-8", newline="\n")
        metadata = {
            "limit": 4, "records_written": 3, "downloaded_count": 3,
            "inspected_count": 4, "rejected_count": 1,
            "rejected_counts": {"invalid_source_url": 1},
            "rejections_file": rejections_path.name,
            "rejections_sha256": hashlib.sha256(rejections_path.read_bytes()).hexdigest(),
        }
        metadata.update(overrides)
        return self.write_acquisition_metadata(**metadata)

    def prepare(self, **overrides):
        arguments = {
            "dataset": "test/source", "revision": REVISION, "license_name": "CC-BY-4.0",
            "train_count": 1, "validation_count": 1, "test_count": 1,
            "min_words": 20, "max_words": 100,
        }
        arguments.update(overrides)
        return prepare_jsonl(self.input_path, self.output_dir, **arguments)

    def read_jsonl(self, name):
        return [
            json.loads(line)
            for line in (self.output_dir / name).read_text(encoding="utf-8").splitlines()
        ]

    def assert_no_partial_output(self):
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(list(self.root.glob(".*.building-*")))

    def test_provenance_hashes_metadata_and_manifest(self):
        records = [make_record(index) for index in range(3)]
        records[0].update(split="upstream_train", quality={"score": 0.94})
        records[0]["source_dataset"] = "untrusted override"
        records[0]["source_id"] = "untrusted override"
        self.write_records(records)
        result = self.prepare()
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(result["scanned_records"], 3)
        self.assertEqual(result["rejected_counts"], {})
        self.assertEqual(
            result["documents_sha256"],
            hashlib.sha256((self.output_dir / "documents.jsonl").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            {"documents.jsonl", "rejections.jsonl", "manifest.json"},
        )
        by_id = {row["source_id"]: row for row in self.read_jsonl("documents.jsonl")}
        for source in records:
            row = by_id[source["id"]]
            self.assertEqual(row["source_url"], source["url"])
            self.assertEqual(row["source_dataset"], "test/source")
            self.assertEqual(row["source_revision"], REVISION)
            self.assertEqual(row["source_license"], "CC-BY-4.0")
            self.assertEqual(row["document_sha256"], hashlib.sha256(source["text"].encode()).hexdigest())
            self.assertEqual(
                row["normalized_sha256"],
                hashlib.sha256(" ".join(source["text"].casefold().split()).encode()).hexdigest(),
            )
        self.assertEqual(by_id["source-0"]["source_split"], "upstream_train")
        self.assertEqual(by_id["source-0"]["quality"], {"score": 0.94})

    def test_determinism_and_seeded_sampling(self):
        self.write_records(make_record(index) for index in range(30))
        first = self.prepare(seed=123)
        documents = (self.output_dir / "documents.jsonl").read_bytes()
        rejections = (self.output_dir / "rejections.jsonl").read_bytes()
        self.output_dir = self.root / "repeat"
        second = self.prepare(seed=123)
        self.assertEqual(documents, (self.output_dir / "documents.jsonl").read_bytes())
        self.assertEqual(rejections, (self.output_dir / "rejections.jsonl").read_bytes())
        self.assertEqual(first["split_counts"], second["split_counts"])
        self.assertEqual(first["rejected_counts"], {"excluded_sampling": 27})
        self.output_dir = self.root / "different"
        self.prepare(seed=456)
        self.assertNotEqual(documents, (self.output_dir / "documents.jsonl").read_bytes())

    def test_exact_and_near_duplicate_groups_before_splitting(self):
        original = make_record("original", words=90)
        exact = {**original, "id": "exact", "text": "  " + original["text"].upper().replace(" ", "\n") + " "}
        near = {**original, "id": "near", "text": original["text"] + " appended"}
        near_exact = {**near, "id": "near-exact"}
        self.write_records([
            original, exact, near, near_exact,
            make_record("other-one", words=90), make_record("other-two", words=90),
        ])
        result = self.prepare()
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(result["rejected_counts"], {"exact_duplicate": 2, "near_duplicate": 1})
        rejected = {row["source_id"]: row for row in self.read_jsonl("rejections.jsonl")}
        self.assertEqual(rejected["exact"]["duplicate_of"], original["id"])
        self.assertEqual(rejected["near"]["duplicate_of"], original["id"])
        self.assertEqual(rejected["near-exact"]["duplicate_of"], "near")
        self.assertEqual(rejected["near-exact"]["representative_source_id"], original["id"])
        rows = self.read_jsonl("documents.jsonl")
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                left_shingles = prepare._shingles(prepare._normalize(left["text"]).split())
                right_shingles = prepare._shingles(prepare._normalize(right["text"]).split())
                self.assertLess(len(left_shingles & right_shingles) / len(left_shingles | right_shingles), 0.8)

    def test_prefix_filter_matches_brute_force_greedy_jaccard(self):
        rng = random.Random(71)
        records = []
        for group in range(12):
            base = [f"g{group}_word{word}" for word in range(55)]
            for variant in range(7):
                words = base[:]
                for position in rng.sample(range(len(words)), variant):
                    words[position] = f"replacement{group}_{variant}_{position}"
                row = make_record(f"{group}-{variant}")
                row["text"] = " ".join(words)
                records.append(row)
        for threshold in (0.3, 0.5, 0.8, 1.0):
            with self.subTest(threshold=threshold):
                retained = []
                for record in records:
                    shingles = prepare._shingles(record["text"].split())
                    if not any(
                        len(shingles & other) / len(shingles | other) >= threshold
                        for _, other in retained
                    ):
                        retained.append((record["id"], shingles))
                self.write_records(records)
                self.output_dir = self.root / f"threshold-{threshold}"
                self.prepare(
                    train_count=100, validation_count=100, test_count=100,
                    near_duplicate_threshold=threshold,
                )
                self.assertEqual(
                    {row["source_id"] for row in self.read_jsonl("documents.jsonl")},
                    {source_id for source_id, _ in retained},
                )

    def test_normalization_handles_unicode_and_whitespace(self):
        original = make_record(1)
        original["text"] = "Ｆｕｌｌｗｉｄｔｈ " + original["text"]
        duplicate = {**original, "id": "normalized-copy", "text": "fullwidth\t" + make_record(1)["text"]}
        self.write_records([original, duplicate])
        result = self.prepare()
        self.assertEqual(result["rejected_counts"], {"exact_duplicate": 1})

    def test_jaccard_threshold_is_inclusive_with_different_lengths(self):
        original = make_record("original", words=8)
        at_threshold = {
            **original, "id": "at-threshold",
            "text": original["text"] + " appended",
        }
        below_threshold = {
            **original, "id": "below-threshold",
            "text": original["text"] + " appended another",
        }
        self.write_records([original, at_threshold, below_threshold])
        result = self.prepare(min_words=1)
        self.assertEqual(result["rejected_counts"], {"near_duplicate": 1})
        self.assertEqual(
            {row["source_id"] for row in self.read_jsonl("documents.jsonl")},
            {original["id"], "below-threshold"},
        )
        self.assertEqual(
            self.read_jsonl("rejections.jsonl")[0]["source_id"], "at-threshold",
        )

    def test_threshold_one_compares_shingle_sets_not_only_text_hash(self):
        original = make_record(1)
        original["text"] = "one two three four five " * 2
        duplicate = {**original, "id": "repeated-cycle", "text": "one two three four five " * 3}
        self.write_records([original, duplicate])
        result = self.prepare(min_words=1, near_duplicate_threshold=1)
        self.assertEqual(result["rejected_counts"], {"near_duplicate": 1})

    def test_documents_under_five_words_use_whole_document_shingle(self):
        rows = [
            {**make_record(1), "text": "two words"},
            {**make_record(2), "text": "two different words"},
            {**make_record(3), "text": "TWO WORDS"},
        ]
        self.write_records(rows)
        result = self.prepare(min_words=1)
        self.assertEqual(result["unique_records"], 2)
        self.assertEqual(result["rejected_counts"], {"exact_duplicate": 1})

    def test_small_positive_threshold_and_large_counts_are_bounded(self):
        self.write_records([make_record(index) for index in range(3)])
        result = self.prepare(
            near_duplicate_threshold=5e-324, train_count=1 << 64,
            max_per_domain=1 << 64,
        )
        self.assertEqual(result["selected_records"], 3)
        self.assertEqual(result["shortfall"], (1 << 64) - 1)

    def test_domain_cap_and_length_strata(self):
        records = [
            make_record(f"{domain}-{length}-{index}", domain=f"{domain}.example", words=length)
            for domain in ("large", "small")
            for length in (25, 55, 85)
            for index in range(5)
        ]
        self.write_records(records)
        result = self.prepare(train_count=2, validation_count=2, test_count=2, max_per_domain=3)
        rows = self.read_jsonl("documents.jsonl")
        self.assertEqual(result["rejected_counts"], {"domain_cap": 24})
        for domain in ("large", "small"):
            selected = [row for row in rows if f"{domain}.example" in row["source_url"]]
            self.assertEqual(sorted(len(row["text"].split()) for row in selected), [25, 55, 85])

    def test_domain_round_robin_prevents_one_domain_dominating(self):
        records = [make_record(index, domain="large.example") for index in range(100)]
        records.extend(make_record(f"small-{index}", domain=f"small{index}.example") for index in range(2))
        self.write_records(records)
        result = self.prepare()
        domains = {row["source_url"].split("/")[2] for row in self.read_jsonl("documents.jsonl")}
        self.assertEqual(domains, {"large.example", "small0.example", "small1.example"})
        self.assertEqual(result["rejected_counts"], {"excluded_sampling": 99})

    def test_shortfall_and_all_scanned_records_accounted_for(self):
        self.write_records([make_record(1), make_record(2)])
        result = self.prepare(train_count=2, validation_count=2, test_count=2)
        self.assertEqual(result["shortfall"], 4)
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 0})
        self.assertEqual(sum(result["split_shortfalls"].values()), 4)
        self.assertEqual(result["scanned_records"], result["selected_records"] + sum(result["rejected_counts"].values()))

    def test_screening_has_explicit_reasons_and_no_rejected_text(self):
        rows = [make_record(index) for index in range(7)]
        rows[0]["text"] = "short"
        rows[1]["text"] = make_record("long", words=101)["text"]
        rows[2]["text"] += " somebody@example.com"
        rows[3]["text"] += " api_key=abcdefghijklmnop123456"
        rows[4]["text"] = "accept all cookies privacy policy " * 5
        rows[5]["text"] = "\n".join(["repeated navigation menu line items"] * 6)
        rows[6]["metadata"] = {"api_key": "abcdefghijklmnop123456"}
        self.write_records(rows)
        result = self.prepare()
        self.assertEqual(result["selected_records"], 0)
        self.assertEqual(result["rejected_counts"], {"source_screening": 7})
        self.assertEqual(
            result["screening_counts"],
            {
                "length_too_short": 1, "length_too_long": 1, "obvious_email": 1,
                "obvious_credential": 2, "boilerplate": 1, "repeated_boilerplate": 1,
            },
        )
        for row in self.read_jsonl("rejections.jsonl"):
            self.assertIn("source_id", row)
            self.assertIn("screening_reason", row)
            self.assertNotIn("text", row)
        self.assertEqual(
            result["documents_sha256"], hashlib.sha256(b"").hexdigest(),
        )

    def test_bounded_scan_does_not_read_or_validate_next_record(self):
        self.write_records(make_record(index) for index in range(3))
        with self.input_path.open("ab") as source:
            source.write(b"malformed fourth row, must not be read")
        result = self.prepare(max_records=3)
        self.assertEqual(result["scanned_records"], 3)
        self.assertTrue(result["scan_limit_reached"])
        self.assertEqual(result["shortfall"], 0)

    def test_invalid_json_and_fields_have_location_not_source_content(self):
        failures = [
            b'{"sensitive-source-content": broken}\n',
            b"\n",
            b"\xff\n",
            b'{"text": "sensitive-source-content", "id": 4, "url": "https://example.com"}\n',
            b'{"text": "sensitive-source-content", "id": "1", "url": "file:///secret"}\n',
            b'{"text": "sensitive-source-content", "id": "1", "url": "https://user:pass@example.com"}\n',
            b'{"text": "sensitive-source-content", "id": "1", "url": "https://example.com:wrong"}\n',
            b'{"text": "sensitive-source-content", "id": "1", "url": "https://example.com", "score": NaN}\n',
            b'{"text": "\\ud800", "id": "1", "url": "https://example.com"}\n',
            b"[]\n",
        ]
        for bad_record in failures:
            with self.subTest(record=bad_record[:30]):
                self.write_records([make_record(0)])
                with self.input_path.open("ab") as source:
                    source.write(bad_record)
                with self.assertRaises(ValueError) as caught:
                    self.prepare()
                self.assertIn(f"{self.input_path}:2:", str(caught.exception))
                self.assertNotIn("sensitive-source-content", str(caught.exception))
                self.assert_no_partial_output()

    def test_oversized_line_fails_without_partial_output(self):
        self.write_records([make_record(0)])
        with patch.object(prepare, "_MAX_LINE_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "safety limit"):
                self.prepare()
        self.assert_no_partial_output()

    def test_parameters_are_validated_before_input_is_opened(self):
        invalid = [
            {"revision": "main"}, {"revision": "a" * 39}, {"revision": "z" * 40},
            {"revision": "sha256:" + "a" * 63}, {"dataset": ""}, {"license_name": " "},
            {"min_words": 30, "max_words": 20}, {"seed": True},
            {"near_duplicate_threshold": 0}, {"near_duplicate_threshold": 1.01},
            {"near_duplicate_threshold": float("nan")},
            {"near_duplicate_threshold": float("inf")},
            {"near_duplicate_threshold": True}, {"near_duplicate_threshold": "0.8"},
        ]
        for name in (
            "train_count", "validation_count", "test_count", "max_records",
            "min_words", "max_words", "max_per_domain", "context_limit",
            "summary_token_reserve",
        ):
            invalid.extend({name: value} for value in (0, -1, True, 1.5))
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.prepare(**overrides)
                self.assert_no_partial_output()

    def test_tokenizer_overflow_screening_precedes_dedup_and_split_assignment(self):
        original = make_record("overflow")
        original["text"] = original["text"].upper()
        just_fit = {**original, "id": "just-fit", "text": original["text"].lower()}
        other = make_record("other")
        self.write_records([original, just_fit, other])
        tokenizer_path = self.root / "tokenizer.json"
        tokenizer_path.write_text('{"fake": "offline tokenizer"}', encoding="utf-8")
        tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        with patch("summary_dataset_builder.tokenization.LocalTokenizer") as tokenizer_class:
            tokenizer_class.return_value.count.side_effect = [1793, 1792, 50]
            result = self.prepare(tokenizer_path=tokenizer_path)
            tokenizer_class.assert_called_once_with(tokenizer_path)
            self.assertEqual(
                [call.args[0] for call in tokenizer_class.return_value.count.call_args_list],
                [training_prompt(record["text"]) for record in (original, just_fit, other)],
            )
        self.assertEqual(result["eligible_records"], 2)
        self.assertEqual(result["unique_records"], 2)
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 0})
        self.assertEqual(result["shortfall"], 1)
        self.assertEqual(result["rejected_counts"], {"source_context_overflow": 1})
        self.assertEqual(
            {row["source_id"] for row in self.read_jsonl("documents.jsonl")},
            {"just-fit", other["id"]},
        )
        rejection = self.read_jsonl("rejections.jsonl")[0]
        self.assertEqual(rejection, {
            "source_id": original["id"], "reason": "source_context_overflow",
            "source_prompt_tokens": 1793, "summary_token_reserve": 256,
            "required_context_tokens": 2049, "context_limit": 2048,
            "tokenizer_sha256": tokenizer_hash,
        })
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["arguments"]["tokenizer_path"], str(tokenizer_path))
        self.assertEqual(manifest["arguments"]["tokenizer_sha256"], tokenizer_hash)
        self.assertEqual(manifest["arguments"]["context_limit"], 2048)
        self.assertEqual(manifest["arguments"]["summary_token_reserve"], 256)
        self.assertEqual(
            manifest["documents_sha256"],
            hashlib.sha256((self.output_dir / "documents.jsonl").read_bytes()).hexdigest(),
        )

    def test_tokenizer_screening_preserves_requested_splits_when_enough_fit(self):
        self.write_records([make_record(index) for index in range(4)])
        tokenizer_path = self.root / "tokenizer.json"
        tokenizer_path.write_text("offline fixture", encoding="utf-8")
        with patch("summary_dataset_builder.tokenization.LocalTokenizer") as tokenizer_class:
            tokenizer_class.return_value.count.side_effect = [81, 80, 79, 20]
            result = self.prepare(
                tokenizer_path=tokenizer_path, context_limit=100, summary_token_reserve=20,
            )
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual(result["rejected_counts"], {"source_context_overflow": 1})

    def test_tokenizer_is_never_loaded_without_opt_in(self):
        self.write_records([make_record(index) for index in range(3)])
        with patch(
            "summary_dataset_builder.tokenization.LocalTokenizer",
            side_effect=AssertionError("must not load tokenizer"),
        ):
            result = self.prepare(context_limit=1, summary_token_reserve=100)
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual(result["selected_records"], 3)
        self.assertIsNone(result["arguments"]["tokenizer_path"])
        self.assertIsNone(result["arguments"]["tokenizer_sha256"])

    def test_tokenizer_must_be_local_and_not_a_secret_file(self):
        self.write_records([make_record(0)])
        with patch("summary_dataset_builder.tokenization.LocalTokenizer") as tokenizer_class:
            with self.assertRaisesRegex(ValueError, "local tokenizer file"):
                self.prepare(tokenizer_path=self.root / "missing-tokenizer.json")
            with self.assertRaisesRegex(ValueError, "secret or credential"):
                self.prepare(tokenizer_path=self.root / "ApiKey.txt")
            tokenizer_class.assert_not_called()
        self.assert_no_partial_output()

    def test_tokenizer_count_failure_is_sanitized_and_atomic(self):
        self.write_records([make_record(0)])
        tokenizer_path = self.root / "tokenizer.json"
        tokenizer_path.write_text("offline fixture", encoding="utf-8")
        with patch("summary_dataset_builder.tokenization.LocalTokenizer") as tokenizer_class:
            tokenizer_class.return_value.count.side_effect = RuntimeError("sensitive source content")
            with self.assertRaisesRegex(ValueError, ":1: local tokenizer failed") as caught:
                self.prepare(tokenizer_path=tokenizer_path)
            self.assertNotIn("sensitive source content", str(caught.exception))
        self.assert_no_partial_output()

    def test_local_sha256_revision_is_verified_against_input_file(self):
        self.write_records([make_record(0)])
        revision = "sha256:" + hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        result = self.prepare(revision=revision)
        self.assertTrue(result["source_revision_verified"])
        self.assertEqual(self.read_jsonl("documents.jsonl")[0]["source_revision"], revision)
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["source_revision_verified"])
        self.assertEqual(
            manifest["documents_sha256"],
            hashlib.sha256((self.output_dir / "documents.jsonl").read_bytes()).hexdigest(),
        )

    def test_local_sha256_mismatch_fails_without_publishing(self):
        self.write_records([make_record(0)])
        with self.assertRaisesRegex(ValueError, "file SHA256 does not match local snapshot revision"):
            self.prepare(revision="sha256:" + "0" * 64)
        self.assert_no_partial_output()

    def test_local_sha256_verifies_bytes_beyond_bounded_record_scan(self):
        self.write_records([make_record(0)])
        prefix_digest = hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        with self.input_path.open("ab") as handle:
            handle.write(b"malformed second record is not parsed\n")
        with self.assertRaisesRegex(ValueError, "does not match local snapshot revision"):
            self.prepare(revision="sha256:" + prefix_digest, max_records=1)
        self.assert_no_partial_output()
        full_digest = hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        result = self.prepare(revision="sha256:" + full_digest.upper(), max_records=1)
        self.assertTrue(result["source_revision_verified"])
        self.assertEqual(result["scanned_records"], 1)

    def test_arbitrary_huggingface_commit_is_not_claimed_verified(self):
        self.write_records([make_record(0)])
        result = self.prepare()
        self.assertFalse(result["source_revision_verified"])

    def test_acquisition_metadata_is_verified_and_retained(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_acquisition_metadata()
        result = self.prepare()
        self.assertTrue(result["acquisition_metadata_verified"])
        self.assertEqual(result["acquisition_metadata"]["dataset"], "test/source")
        self.assertEqual(result["acquisition_metadata"]["revision"], REVISION)
        self.assertEqual(result["acquisition_metadata"]["subset"], "sample")
        self.assertEqual(result["shortfall"], 0)
        self.assertFalse(result["source_revision_verified"])

    def test_audited_acquisition_counts_distinguish_inspected_and_emitted_rows(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_audited_acquisition_metadata()
        result = self.prepare()
        self.assertTrue(result["acquisition_metadata_verified"])
        self.assertEqual(result["scanned_records"], 3)
        self.assertEqual(result["selected_records"], 3)
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual(result["acquisition_metadata"]["inspected_count"], 4)
        self.assertEqual(result["acquisition_metadata"]["downloaded_count"], 3)
        self.assertEqual(result["acquisition_metadata"]["rejected_count"], 1)
        self.assertEqual(result["acquisition_metadata"]["rejected_counts"], {"invalid_source_url": 1})
        self.output_dir = self.root / "bounded-prepared"
        result = self.prepare(max_records=2)
        self.assertEqual(result["scanned_records"], 2)
        self.assertEqual(result["shortfall"], 1)

    def test_audited_acquisition_rejects_inconsistent_counters(self):
        self.write_records([make_record(index) for index in range(3)])
        invalid = [
            {"inspected_count": 3}, {"inspected_count": 5},
            {"downloaded_count": 2}, {"downloaded_count": True},
            {"rejected_count": -1}, {"rejected_count": 2},
            {"rejected_counts": {"invalid_source_url": 2}},
            {"rejected_counts": {"invalid_source_url": True}},
            {"rejected_counts": {"invalid_source_url": 0}},
            {"rejected_counts": {"unsafe reason": 1}},
            {"rejected_counts": None}, {"rejections_sha256": "wrong"},
            {"rejections_file": "unrelated.jsonl"},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                self.write_audited_acquisition_metadata(**overrides)
                with self.assertRaisesRegex(ValueError, "acquisition"):
                    self.prepare()
                self.assert_no_partial_output()
        path = self.write_audited_acquisition_metadata()
        metadata = json.loads(path.read_text(encoding="utf-8"))
        del metadata["downloaded_count"]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete acquisition rejection metadata"):
            self.prepare()
        self.assert_no_partial_output()

    def test_acquisition_rejection_sidecar_is_required_and_checksum_verified(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_audited_acquisition_metadata()
        rejections_path = self.input_path.with_name(self.input_path.name + ".rejections.jsonl")
        with rejections_path.open("ab") as handle:
            handle.write(b"tampered audit")
        with self.assertRaisesRegex(ValueError, "rejection sidecar SHA256 does not match"):
            self.prepare()
        self.assert_no_partial_output()
        self.write_audited_acquisition_metadata()
        rejections_path.unlink()
        with self.assertRaisesRegex(ValueError, "rejection sidecar is missing"):
            self.prepare()
        self.assert_no_partial_output()

    def test_acquisition_rejection_sidecar_count_must_match_metadata(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_audited_acquisition_metadata(
            inspected_count=3, rejected_count=0, rejected_counts={},
        )
        with self.assertRaisesRegex(ValueError, "rejection sidecar count does not match"):
            self.prepare()
        self.assert_no_partial_output()

    def test_acquisition_metadata_rejects_dataset_and_revision_overrides(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_acquisition_metadata()
        for overrides, field in (
            ({"dataset": "different/source"}, "dataset"),
            ({"revision": "c" * 40}, "revision"),
            ({"revision": "sha256:" + "a" * 64}, "revision"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} does not match acquisition metadata"):
                    self.prepare(**overrides)
                self.assert_no_partial_output()

    def test_acquisition_metadata_rejects_changed_snapshot(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_acquisition_metadata()
        with self.input_path.open("ab") as handle:
            handle.write(json.dumps(make_record(4)).encode("utf-8") + b"\n")
        with self.assertRaisesRegex(ValueError, "file SHA256 does not match"):
            self.prepare(max_records=2)
        self.assert_no_partial_output()

    def test_acquisition_hash_verification_does_not_parse_beyond_record_limit(self):
        self.write_records([make_record(index) for index in range(2)])
        with self.input_path.open("ab") as handle:
            handle.write(b"third record is not valid JSON\n")
        self.write_acquisition_metadata()
        result = self.prepare(max_records=2)
        self.assertEqual(result["scanned_records"], 2)
        self.assertEqual(result["shortfall"], 1)
        self.assertTrue(result["acquisition_metadata_verified"])

    def test_invalid_acquisition_metadata_fails_explicitly(self):
        self.write_records([make_record(index) for index in range(3)])
        invalid = [
            {"format_version": True}, {"format_version": 2},
            {"revision": "main"}, {"download_sha256": "invalid"},
            {"limit": 0}, {"records_written": -1}, {"records_written": 4},
            {"records_written": True}, {"subset": ""}, {"dataset": None},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                self.write_acquisition_metadata(**overrides)
                with self.assertRaisesRegex(ValueError, "acquisition metadata"):
                    self.prepare()
                self.assert_no_partial_output()
        path = self.write_acquisition_metadata()
        path.write_bytes(b"invalid sensitive-metadata-content")
        with self.assertRaisesRegex(ValueError, "invalid acquisition metadata JSON") as caught:
            self.prepare()
        self.assertNotIn("sensitive-metadata-content", str(caught.exception))
        self.assert_no_partial_output()

    def test_acquisition_metadata_rejects_inconsistent_counts_and_source_split(self):
        self.write_records([make_record(index) for index in range(3)])
        self.write_acquisition_metadata(records_written=2)
        with self.assertRaisesRegex(ValueError, "record count does not match"):
            self.prepare()
        self.assert_no_partial_output()
        records = [make_record(index) for index in range(3)]
        records[1]["source_split"] = "validation"
        self.write_records(records)
        self.write_acquisition_metadata()
        with self.assertRaisesRegex(ValueError, ":2: source_split does not match"):
            self.prepare()
        self.assert_no_partial_output()

    def test_secret_input_paths_are_refused_without_opening(self):
        for name in ("ApiKey.txt", ".env", ".env.local", "credentials.json", "private.pem"):
            with self.subTest(name=name):
                self.input_path = self.root / name
                with patch.object(Path, "open", side_effect=AssertionError("must not read")):
                    with self.assertRaisesRegex(ValueError, "secret or credential"):
                        self.prepare()

    def test_existing_output_is_preserved(self):
        self.write_records([make_record(0)])
        self.output_dir.mkdir()
        marker = self.output_dir / "existing"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_empty_existing_output_is_also_refused(self):
        self.write_records([make_record(0)])
        self.output_dir.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(list(self.output_dir.iterdir()), [])

    def test_resolved_secret_path_is_refused(self):
        self.input_path = self.root / "apparently-safe.jsonl"
        with patch.object(Path, "resolve", return_value=self.root / "ApiKey.txt"):
            with patch.object(Path, "open", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "secret or credential"):
                    self.prepare()

    def test_empty_input_is_an_explicit_shortfall(self):
        self.write_records([])
        result = self.prepare()
        self.assertEqual(result["scanned_records"], 0)
        self.assertEqual(result["shortfall"], 3)
        self.assertEqual(result["split_counts"], {"train": 0, "validation": 0, "test": 0})

    def test_write_failure_cleans_unpublished_work(self):
        self.write_records([make_record(0)])
        with patch.object(prepare.os, "fsync", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                self.prepare()
        self.assert_no_partial_output()

    def test_ten_thousand_records_use_disk_index_and_complete(self):
        self.write_records(
            make_record(index, domain=f"domain{index % 50}.example", words=20)
            for index in range(10000)
        )
        result = self.prepare(
            train_count=9800, validation_count=100, test_count=100, max_records=10000,
        )
        self.assertEqual(result["selected_records"], 10000)
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual(result["rejected_counts"], {})

    def test_preserved_records_and_metadata_are_exact_and_extension_is_deterministic(self):
        anchors = [
            make_anchor("anchor-train", split="train"),
            make_anchor("anchor-validation", split="validation"),
            make_anchor("anchor-test", split="test"),
        ]
        preserve_path = self.write_preserved(anchors)
        preserved_bytes = preserve_path.read_bytes()
        incoming = [make_record(index) for index in range(20)]
        incoming.extend(
            {"id": anchor["source_id"], "url": anchor["source_url"],
             "text": anchor["text"], "split": "raw-upstream-split"}
            for anchor in anchors
        )
        self.write_records(incoming)
        arguments = {"preserve_path": preserve_path, "train_count": 3, "validation_count": 2, "test_count": 2}
        first = self.prepare(**arguments)
        documents = (self.output_dir / "documents.jsonl").read_bytes()
        rejections = (self.output_dir / "rejections.jsonl").read_bytes()
        self.assertTrue(documents.startswith(preserved_bytes))
        self.assertEqual(self.read_jsonl("documents.jsonl")[:3], anchors)
        self.assertEqual(first["preserved_count"], 3)
        self.assertEqual(first["preserved_split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(first["preserved_documents_sha256"], hashlib.sha256(preserved_bytes).hexdigest())
        self.assertEqual(first["arguments"]["preserve_path"], str(preserve_path))
        self.assertEqual(first["new_selected_records"], 4)
        self.assertEqual(first["selected_records"], 7)
        self.assertEqual(first["split_counts"], {"train": 3, "validation": 2, "test": 2})
        self.assertEqual(first["shortfall"], 0)
        self.assertEqual(first["rejected_counts"], {"existing_source": 3, "excluded_sampling": 16})
        self.assertEqual(
            first["scanned_records"],
            first["new_selected_records"] + sum(first["rejected_counts"].values()),
        )
        self.assertTrue(first["global_dedup_invariants"]["anchors_indexed_first"])
        self.assertTrue(first["global_dedup_invariants"]["unique_source_ids"])
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["preserved_documents_sha256"], hashlib.sha256(preserved_bytes).hexdigest())
        self.assertEqual(manifest["documents_sha256"], hashlib.sha256(documents).hexdigest())
        self.output_dir = self.root / "repeat-preservation"
        second = self.prepare(**arguments)
        self.assertEqual(documents, (self.output_dir / "documents.jsonl").read_bytes())
        self.assertEqual(rejections, (self.output_dir / "rejections.jsonl").read_bytes())
        self.assertEqual(first["split_counts"], second["split_counts"])
        self.assertEqual(preserve_path.read_bytes(), preserved_bytes)

    def test_preserved_records_absent_from_input_still_fill_requested_slots(self):
        anchors = [
            make_anchor("train", split="train"), make_anchor("validation", split="validation"),
            make_anchor("test", split="test"),
        ]
        preserve_path = self.write_preserved(anchors, trailing_newline=False)
        self.write_records([make_record(index) for index in range(5)])
        result = self.prepare(preserve_path=preserve_path)
        self.assertEqual(self.read_jsonl("documents.jsonl"), anchors)
        self.assertEqual(result["selected_records"], 3)
        self.assertEqual(result["new_selected_records"], 0)
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual(result["rejected_counts"], {"excluded_sampling": 5})
        self.assertEqual(
            result["preserved_documents_sha256"], hashlib.sha256(preserve_path.read_bytes()).hexdigest(),
        )

    def test_exact_and_near_duplicates_of_every_anchor_split_are_excluded(self):
        anchors = [
            make_anchor(f"anchor-{split}", split=split, words=90)
            for split in ("train", "validation", "test")
        ]
        preserve_path = self.write_preserved(anchors)
        incoming = [
            {"id": f"near-{index}", "url": f"https://near{index}.example/article",
             "text": anchor["text"] + " appended"}
            for index, anchor in enumerate(anchors)
        ]
        incoming.append({
            "id": "exact-anchor-copy", "url": "https://exact.example/article",
            "text": anchors[2]["text"].upper().replace(" ", "\n"),
        })
        incoming.extend([make_record("novel-a", words=90), make_record("novel-b", words=90)])
        self.write_records(incoming)
        result = self.prepare(preserve_path=preserve_path, train_count=2, validation_count=2, test_count=2)
        self.assertEqual(result["rejected_counts"], {"near_duplicate": 3, "exact_duplicate": 1})
        self.assertEqual(result["selected_records"], 5)
        self.assertEqual(result["shortfall"], 1)
        self.assertEqual(result["split_counts"], {"train": 2, "validation": 2, "test": 1})
        rejected = self.read_jsonl("rejections.jsonl")
        for index in range(3):
            self.assertEqual(rejected[index]["duplicate_of"], anchors[index]["source_id"])
        rows = self.read_jsonl("documents.jsonl")
        self.assertEqual(rows[:3], anchors)
        for index, row in enumerate(rows):
            left = prepare._shingles(prepare._normalize(row["text"]).split())
            for other in rows[index + 1:]:
                right = prepare._shingles(prepare._normalize(other["text"]).split())
                self.assertLess(len(left & right) / len(left | right), 0.8)

    def test_anchor_reencounter_changes_fail_before_quality_screening(self):
        anchor = make_anchor("original")
        preserve_path = self.write_preserved([anchor])
        conflicting = [
            {"text": "too short"},
            {"text": anchor["text"].upper()},
            {"url": "https://different.example/article"},
        ]
        for changes in conflicting:
            with self.subTest(changes=tuple(changes)):
                row = {"id": anchor["source_id"], "url": anchor["source_url"], "text": anchor["text"]}
                row.update(changes)
                self.write_records([row])
                with self.assertRaisesRegex(ValueError, ":1: source_id reencounter has different content or source URL"):
                    self.prepare(preserve_path=preserve_path)
                self.assert_no_partial_output()

    def test_conflicting_new_source_ids_are_fatal(self):
        preserve_path = self.write_preserved([make_anchor("original")])
        first = make_record("new")
        second = {**make_record("changed"), "id": first["id"]}
        self.write_records([first, second])
        with self.assertRaisesRegex(ValueError, ":2: source_id reencounter"):
            self.prepare(preserve_path=preserve_path)
        self.assert_no_partial_output()

    def test_preserved_domain_occupancy_limits_new_sources(self):
        anchor = make_anchor("old", domain="occupied.example")
        preserve_path = self.write_preserved([anchor])
        self.write_records([
            make_record("same-domain", domain="www.occupied.example"),
            make_record("new-domain-a"), make_record("new-domain-b"),
        ])
        result = self.prepare(preserve_path=preserve_path, max_per_domain=1)
        self.assertEqual(result["rejected_counts"], {"domain_cap": 1})
        self.assertEqual(result["selected_records"], 3)
        self.assertEqual(self.read_jsonl("documents.jsonl")[0], anchor)

    def test_preserved_length_occupancy_guides_remaining_diversity_slots(self):
        anchor = make_anchor("short-anchor", domain="one.example", words=25)
        preserve_path = self.write_preserved([anchor])
        self.write_records(
            make_record(f"new-{length}-{index}", domain="one.example", words=length)
            for length in (25, 55, 85) for index in range(5)
        )
        result = self.prepare(preserve_path=preserve_path, max_per_domain=3)
        self.assertEqual(result["rejected_counts"], {"domain_cap": 13})
        self.assertEqual(
            sorted(len(row["text"].split()) for row in self.read_jsonl("documents.jsonl")),
            [25, 55, 85],
        )
        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})

    def test_preserved_records_are_not_dropped_by_new_length_screening(self):
        anchor = make_anchor("short-anchor", split="test", words=25)
        preserve_path = self.write_preserved([anchor])
        self.write_records([make_record("short-anchor", words=25)])
        result = self.prepare(preserve_path=preserve_path, min_words=50)
        self.assertEqual(self.read_jsonl("documents.jsonl"), [anchor])
        self.assertEqual(result["rejected_counts"], {"existing_source": 1})
        self.assertEqual(result["shortfall"], 2)
        self.assertEqual(result["split_shortfalls"], {"train": 1, "validation": 1, "test": 0})

    def test_preserved_context_overflow_is_fatal_not_a_silent_removal(self):
        preserve_path = self.write_preserved([make_anchor("original")])
        self.write_records([])
        tokenizer_path = self.root / "tokenizer.json"
        tokenizer_path.write_text("offline fixture", encoding="utf-8")
        with patch("summary_dataset_builder.tokenization.LocalTokenizer") as tokenizer_class:
            tokenizer_class.return_value.count.return_value = 1793
            with self.assertRaisesRegex(ValueError, "preserved source_context_overflow"):
                self.prepare(preserve_path=preserve_path, tokenizer_path=tokenizer_path)
        self.assert_no_partial_output()

    def test_bad_anchor_schema_hashes_and_provenance_fail_atomically(self):
        self.write_records([])
        bad_changes = [
            {"source_id": 7}, {"source_id": ""}, {"source_url": "ftp://example.org"},
            {"source_dataset": "another/source"}, {"source_revision": "b" * 40},
            {"source_license": "different-license"}, {"split": "dev"},
            {"document_sha256": "0" * 64}, {"normalized_sha256": "0" * 64},
            {"normalized_sha256": None}, {"text": ""}, {"text": "\ud800"},
            {"extra": float("nan")},
        ]
        for changes in bad_changes:
            with self.subTest(fields=tuple(changes)):
                anchor = {**make_anchor("original"), **changes}
                preserve_path = self.write_preserved([anchor])
                with self.assertRaises(ValueError) as caught:
                    self.prepare(preserve_path=preserve_path)
                self.assertIn(f"{preserve_path}:1:", str(caught.exception))
                self.assert_no_partial_output()
        anchor = make_anchor("missing-hash")
        del anchor["document_sha256"]
        for bad in (anchor, [], None):
            preserve_path = self.write_preserved([bad])
            with self.assertRaises(ValueError):
                self.prepare(preserve_path=preserve_path)
            self.assert_no_partial_output()

    def test_bad_anchor_json_has_location_without_source_content(self):
        self.write_records([])
        preserve_path = self.root / "anchors.jsonl"
        for raw in (b'{"private-source-text": broken}\n', b"\xff\n", b"\n"):
            preserve_path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "invalid preserved UTF-8 JSON record") as caught:
                self.prepare(preserve_path=preserve_path)
            self.assertIn(f"{preserve_path}:1:", str(caught.exception))
            self.assertNotIn("private-source-text", str(caught.exception))
            self.assert_no_partial_output()

    def test_ambiguous_duplicate_json_fields_in_anchors_are_rejected(self):
        self.write_records([])
        preserve_path = self.root / "anchors.jsonl"
        raw = json.dumps(make_anchor("original"))[:-1] + ', "source_id": "duplicate-field"}\n'
        preserve_path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid preserved UTF-8 JSON record"):
            self.prepare(preserve_path=preserve_path)
        self.assert_no_partial_output()

    def test_duplicate_anchor_ids_exact_text_and_near_text_are_rejected(self):
        self.write_records([])
        original = make_anchor("original", words=90)
        cases = [
            ({**original, "split": "validation"}, "duplicate preserved source_id"),
            ({**original, "source_id": "copy", "split": "validation"}, "exact_duplicate"),
        ]
        near = {**original, "source_id": "near", "split": "validation", "text": original["text"] + " appended"}
        near["document_sha256"] = hashlib.sha256(near["text"].encode("utf-8")).hexdigest()
        near["normalized_sha256"] = hashlib.sha256(prepare._normalize(near["text"]).encode("utf-8")).hexdigest()
        cases.append((near, "near_duplicate"))
        for other, expected in cases:
            with self.subTest(expected=expected):
                preserve_path = self.write_preserved([original, other])
                with self.assertRaisesRegex(ValueError, expected):
                    self.prepare(preserve_path=preserve_path)
                self.assert_no_partial_output()

    def test_anchor_split_and_domain_caps_cannot_be_silently_exceeded(self):
        self.write_records([])
        preserve_path = self.write_preserved([make_anchor("first"), make_anchor("second")])
        with self.assertRaisesRegex(ValueError, "preserved split count exceeds requested 'train'"):
            self.prepare(preserve_path=preserve_path)
        self.assert_no_partial_output()
        preserve_path = self.write_preserved([
            make_anchor("first", domain="one.example"),
            make_anchor("second", split="validation", domain="one.example"),
        ])
        with self.assertRaisesRegex(ValueError, "preserved domain count exceeds"):
            self.prepare(preserve_path=preserve_path, max_per_domain=1)
        self.assert_no_partial_output()

    def test_missing_or_secret_preservation_path_is_refused(self):
        self.write_records([])
        with self.assertRaises(FileNotFoundError):
            self.prepare(preserve_path=self.root / "missing.jsonl")
        with patch.object(Path, "open", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "secret or credential"):
                self.prepare(preserve_path=self.root / "ApiKey.txt")
        self.assert_no_partial_output()

    def test_empty_preservation_file_and_optional_normalized_hash(self):
        self.write_records([make_record("novel")])
        preserve_path = self.write_preserved([])
        result = self.prepare(preserve_path=preserve_path)
        self.assertEqual(result["preserved_count"], 0)
        self.assertEqual(result["preserved_documents_sha256"], hashlib.sha256(b"").hexdigest())
        anchor = make_anchor("original")
        del anchor["normalized_sha256"]
        self.output_dir = self.root / "optional-normalized-hash"
        preserve_path = self.write_preserved([anchor])
        self.prepare(preserve_path=preserve_path)
        self.assertEqual(self.read_jsonl("documents.jsonl")[0], anchor)

    def test_four_hundred_anchors_survive_a_bounded_expansion(self):
        anchors = [
            make_anchor(
                index, domain=f"domain{index % 50}.example",
                split="train" if index < 300 else "validation" if index < 350 else "test",
            )
            for index in range(400)
        ]
        preserve_path = self.write_preserved(anchors)
        self.write_records(
            make_record(index, domain=f"domain{index % 50}.example") for index in range(1000)
        )
        result = self.prepare(
            preserve_path=preserve_path, train_count=15000, validation_count=400,
            test_count=400, max_records=420,
        )
        self.assertEqual(result["preserved_count"], 400)
        self.assertEqual(result["preserved_split_counts"], {"train": 300, "validation": 50, "test": 50})
        self.assertEqual(result["scanned_records"], 420)
        self.assertEqual(result["new_selected_records"], 20)
        self.assertEqual(result["selected_records"], 420)
        self.assertEqual(result["rejected_counts"], {"existing_source": 400})
        self.assertEqual(result["shortfall"], 15380)
        self.assertEqual(self.read_jsonl("documents.jsonl")[:400], anchors)


if __name__ == "__main__":
    unittest.main()
