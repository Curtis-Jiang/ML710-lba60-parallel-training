from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sequence_binding.config import resolve_repo_path
from sequence_binding.data.batch import PairBatch
from sequence_binding.data.tokenizer import CharTokenizer


class PairBindingTSVDataset(Dataset):
    """Simple in-memory TSV dataset for compact course data."""

    def __init__(
        self,
        *,
        tsv_path: str | Path,
        protein_tokenizer: CharTokenizer,
        smiles_tokenizer: CharTokenizer,
        max_protein_len: int,
        max_smiles_len: int,
    ) -> None:
        self.tsv_path = resolve_repo_path(tsv_path)
        self.protein_tokenizer = protein_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
        self.max_protein_len = int(max_protein_len)
        self.max_smiles_len = int(max_smiles_len)
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "protein_sequence": str(row["protein_sequence"]),
                        "smiles": str(row["smiles"]),
                        "label": float(row["label"]),
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "protein_tokens": self.protein_tokenizer.encode(
                row["protein_sequence"], self.max_protein_len
            ),
            "smiles_tokens": self.smiles_tokenizer.encode(
                row["smiles"], self.max_smiles_len
            ),
            "label": float(row["label"]),
        }


def make_collate_fn(protein_pad_id: int, smiles_pad_id: int):
    def collate(items: list[dict[str, Any]]) -> PairBatch:
        protein_len = max(len(item["protein_tokens"]) for item in items)
        smiles_len = max(len(item["smiles_tokens"]) for item in items)
        batch_size = len(items)

        protein_tokens = torch.full(
            (batch_size, protein_len), protein_pad_id, dtype=torch.long
        )
        protein_mask = torch.zeros((batch_size, protein_len), dtype=torch.bool)
        smiles_tokens = torch.full(
            (batch_size, smiles_len), smiles_pad_id, dtype=torch.long
        )
        smiles_mask = torch.zeros((batch_size, smiles_len), dtype=torch.bool)
        labels = torch.zeros((batch_size,), dtype=torch.float32)

        for row, item in enumerate(items):
            protein = torch.as_tensor(item["protein_tokens"], dtype=torch.long)
            smiles = torch.as_tensor(item["smiles_tokens"], dtype=torch.long)
            protein_tokens[row, : protein.shape[0]] = protein
            protein_mask[row, : protein.shape[0]] = True
            smiles_tokens[row, : smiles.shape[0]] = smiles
            smiles_mask[row, : smiles.shape[0]] = True
            labels[row] = float(item["label"])

        return PairBatch(
            protein_tokens=protein_tokens,
            protein_mask=protein_mask,
            smiles_tokens=smiles_tokens,
            smiles_mask=smiles_mask,
            labels=labels,
        )

    return collate


def build_dataloader(
    dataset: PairBindingTSVDataset,
    *,
    batch_size: int,
    protein_pad_id: int,
    smiles_pad_id: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=make_collate_fn(protein_pad_id, smiles_pad_id),
    )
    return loader, sampler
