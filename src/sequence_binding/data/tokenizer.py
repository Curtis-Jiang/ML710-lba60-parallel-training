from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sequence_binding.config import resolve_repo_path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class CharTokenizer:
    stoi: dict[str, int]

    @property
    def pad_id(self) -> int:
        return int(self.stoi[PAD_TOKEN])

    @property
    def unk_id(self) -> int:
        return int(self.stoi[UNK_TOKEN])

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        normalized = (text or "").strip()
        ids = [self.stoi.get(ch, self.unk_id) for ch in normalized[:max_len]]
        if not ids:
            ids = [self.unk_id]
        return ids


def load_char_tokenizer(path: str | Path) -> CharTokenizer:
    vocab_path = resolve_repo_path(path)
    with vocab_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CharTokenizer(stoi={str(k): int(v) for k, v in payload["stoi"].items()})
