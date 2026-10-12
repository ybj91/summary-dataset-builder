import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from summary_dataset_builder.tokenization import LocalTokenizer
from tests.helpers import TOKENIZER


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "Optional tokenizer extra not installed")
class TokenizationTests(unittest.TestCase):
    def test_exact_local_count_disables_truncation_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            config = dict(TOKENIZER)
            config["truncation"] = {
                "direction": "Right", "max_length": 2, "strategy": "LongestFirst", "stride": 0
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            tokenizer = LocalTokenizer(path)
            self.assertEqual(tokenizer.count("one two three four five"), 5)

    def test_missing_file_does_not_download(self):
        with self.assertRaisesRegex(ValueError, "never downloaded"):
            LocalTokenizer(Path("does-not-exist-tokenizer.json"))
