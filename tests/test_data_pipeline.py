from __future__ import annotations

import json
from pathlib import Path

from sequence_binding.data import PairBindingTSVDataset, build_dataloader, load_char_tokenizer


def _write_vocab(path: Path, chars: list[str]) -> None:
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for idx, ch in enumerate(chars, start=2):
        stoi[ch] = idx
    path.write_text(json.dumps({"stoi": stoi}), encoding="utf-8")


def _build_tiny_dataset(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    protein_vocab = root / "protein_vocab.json"
    smiles_vocab = root / "smiles_vocab.json"
    train_tsv = root / "train.tsv"
    _write_vocab(protein_vocab, ["A", "C", "D", "X"])
    _write_vocab(smiles_vocab, ["C", "N", "O", "="])
    train_tsv.write_text(
        "\n".join(
            [
                "sample_id\tprotein_sequence\tsmiles\tlabel",
                "row-0\tACD\tCCO\t0",
                "row-1\tDXXA\tC=NO\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return train_tsv, protein_vocab, smiles_vocab


def test_tsv_dataset_normalizes_sequence_and_smiles(tmp_path: Path) -> None:
    train_tsv, protein_vocab, smiles_vocab = _build_tiny_dataset(tmp_path)
    dataset = PairBindingTSVDataset(
        tsv_path=train_tsv,
        protein_tokenizer=load_char_tokenizer(protein_vocab),
        smiles_tokenizer=load_char_tokenizer(smiles_vocab),
        max_protein_len=3,
        max_smiles_len=4,
    )
    item = dataset[1]
    assert item["protein_tokens"] == [4, 5, 5]
    assert item["smiles_tokens"] == [2, 5, 3, 4]
    assert item["label"] == 1.0


def test_dataloader_builds_expected_masks(tmp_path: Path) -> None:
    train_tsv, protein_vocab, smiles_vocab = _build_tiny_dataset(tmp_path)
    dataset = PairBindingTSVDataset(
        tsv_path=train_tsv,
        protein_tokenizer=load_char_tokenizer(protein_vocab),
        smiles_tokenizer=load_char_tokenizer(smiles_vocab),
        max_protein_len=8,
        max_smiles_len=8,
    )
    loader, _ = build_dataloader(
        dataset,
        batch_size=2,
        protein_pad_id=0,
        smiles_pad_id=0,
        num_workers=0,
        pin_memory=False,
        distributed=False,
        rank=0,
        world_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert tuple(batch.protein_tokens.shape) == (2, 4)
    assert tuple(batch.smiles_tokens.shape) == (2, 4)
    assert batch.protein_mask.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert batch.smiles_mask.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert batch.labels.tolist() == [0.0, 1.0]
