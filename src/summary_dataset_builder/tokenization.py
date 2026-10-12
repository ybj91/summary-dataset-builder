from pathlib import Path


TRAINING_INSTRUCTION = "Summarize the following document faithfully in one concise paragraph."


def training_prompt(text: str) -> str:
    return f"{TRAINING_INSTRUCTION}\n\nDocument:\n{text}\n\nSummary:\n"


class LocalTokenizer:
    def __init__(self, path: Path):
        if not path.is_file():
            raise ValueError("A local tokenizer.json is required; tokenizers are never downloaded automatically")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ValueError('Install the tokenizer extra: python -m pip install -e ".[tokenizer]"') from None
        try:
            self.tokenizer = Tokenizer.from_file(str(path))
        except Exception:
            # The Rust binding raises an untyped exception and can include the file's contents.
            raise ValueError("Could not load the local tokenizer.json") from None
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)
