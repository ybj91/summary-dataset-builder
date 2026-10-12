import json
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from summary_dataset_builder.files import sha256_file, text_hash
from summary_dataset_builder.store import initialize_run, RunStore


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = {
    "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
    "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None, "decoder": None,
    "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
}


def completion(content=None, *, finish_reason="stop", cost="0.0001"):
    if content is None:
        content = json.dumps({
            "accepted": True, "summary": "The town greenhouse supplies vegetables to local schools.",
            "rejection_reason": "",
        })
    return SimpleNamespace(
        content=content, finish_reason=finish_reason, request_id="test-request",
        model="openai/gpt-oss-120b", provider="test-provider", prompt_tokens=100,
        completion_tokens=50, cost_usd=Decimal(cost) if cost is not None else None, usage={},
    )


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses is not None else None
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses is None:
            return completion()
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeTokenizer:
    def __init__(self, path):
        pass

    def count(self, text):
        return len(text.split())


class RunCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.run = self.root / "run"
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text(json.dumps(TOKENIZER), encoding="utf-8")
        self.write_sources()

    def write_sources(self, count_per_split=4):
        records = []
        for split in ("train", "validation", "test"):
            for index in range(count_per_split):
                text = (
                    f"The town greenhouse supplies vegetables to local schools. "
                    f"This is synthetic source {split} {index}. Volunteers tend its gardens."
                )
                records.append({
                    "source_id": f"{split}-{index}", "source_url": f"https://{split}.example/{index}",
                    "source_dataset": "synthetic-test", "source_revision": "a" * 40,
                    "source_license": "CC0-1.0", "document_sha256": text_hash(text),
                    "text": text, "split": split,
                })
        (self.prepared / "documents.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        (self.prepared / "manifest.json").write_text(json.dumps({
            "format_version": 1, "documents_sha256": sha256_file(self.prepared / "documents.jsonl")
        }), encoding="utf-8")
        return records

    def initialize(self, **overrides):
        from unittest.mock import patch

        kwargs = {
            "model": "openai/gpt-oss-120b", "providers": ["test-provider"], "budget_usd": "1",
            "input_price_per_million": "1", "output_price_per_million": "2",
            "prompt_path": ROOT / "prompts" / "teacher-v2.txt",
            "schema_path": ROOT / "prompts" / "teacher-v2.schema.json",
            "tokenizer_path": self.tokenizer,
            "target_train": 2, "target_validation": 1, "target_test": 1,
        }
        kwargs.update(overrides)
        with patch("summary_dataset_builder.store.LocalTokenizer", FakeTokenizer):
            return initialize_run(self.prepared, self.run, **kwargs)

    def store(self):
        store = RunStore(self.run)
        self.addCleanup(store.db.close)
        return store
