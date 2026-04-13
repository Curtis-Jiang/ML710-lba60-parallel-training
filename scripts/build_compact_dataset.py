#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequence_binding.data.tokenizer import PAD_TOKEN, UNK_TOKEN


DEFAULT_EXTERNAL_TRAIN_TSV = (
    "/jfs/canjiang/gfm/New_Version_Of_pairwise/data/processed/banks/"
    "pubchem_binary_pairagg_fullseq/train/row_metadata.tsv"
)
DEFAULT_EXTERNAL_TRAIN_LABELS = (
    "/jfs/canjiang/gfm/New_Version_Of_pairwise/data/processed/banks/"
    "pubchem_binary_pairagg_fullseq/train/labels.npy"
)
DEFAULT_EXTERNAL_VAL_CSV = (
    "/jfs/canjiang/gfm/New_Version_Of_pairwise/data/processed/banks/"
    "pubchem_binary_pairagg_fullseq_valsampled10x/pubchem_val_sampled10x.csv"
)


def stratified_indices(labels: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels_i = labels.astype(np.int64, copy=False)
    pos_idx = np.flatnonzero(labels_i == 1)
    neg_idx = np.flatnonzero(labels_i == 0)
    pos_target = int(round(target_size * float(labels_i.mean())))
    pos_target = min(max(pos_target, 1), len(pos_idx), target_size - 1)
    neg_target = min(target_size - pos_target, len(neg_idx))
    selected = np.concatenate(
        [
            rng.choice(pos_idx, size=pos_target, replace=False),
            rng.choice(neg_idx, size=neg_target, replace=False),
        ]
    )
    selected.sort()
    return selected.astype(np.int64)


def build_vocab(strings: list[str]) -> dict[str, int]:
    chars = sorted({ch for text in strings for ch in text.strip()})
    stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, ch in enumerate(chars, start=2):
        stoi[ch] = idx
    return stoi


def _read_train_rows(path: Path, keep_indices: set[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row_index = int(row["row_index"])
            if row_index not in keep_indices:
                continue
            rows.append(
                {
                    "sample_id": f"train-{row_index}",
                    "protein_sequence": str(row["protein_sequence"]).strip(),
                    "smiles": str(row["smiles"]).strip(),
                    "label": str(int(float(row["label"]))),
                }
            )
    return rows


def _read_val_rows(path: Path, keep_indices: set[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index not in keep_indices:
                continue
            rows.append(
                {
                    "sample_id": f"val-{row_index}",
                    "protein_sequence": str(row["protein_sequence"]).strip(),
                    "smiles": str(row["smiles"]).strip(),
                    "label": str(int(float(row["label"]))),
                }
            )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "protein_sequence", "smiles", "label"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact TSV dataset for the ML710 course project.")
    parser.add_argument(
        "--output-dir",
        default="data/datasets/pubchem_course",
        help="Output dataset directory relative to the repo root.",
    )
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--val-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-row-metadata", default=DEFAULT_EXTERNAL_TRAIN_TSV)
    parser.add_argument("--train-labels", default=DEFAULT_EXTERNAL_TRAIN_LABELS)
    parser.add_argument("--val-csv", default=DEFAULT_EXTERNAL_VAL_CSV)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    train_tsv = output_dir / "train.tsv"
    val_tsv = output_dir / "val.tsv"
    protein_vocab_json = output_dir / "protein_vocab.json"
    smiles_vocab_json = output_dir / "smiles_vocab.json"
    dataset_meta_json = output_dir / "dataset_meta.json"
    expected_outputs = [train_tsv, val_tsv, protein_vocab_json, smiles_vocab_json, dataset_meta_json]
    if all(path.exists() for path in expected_outputs) and not args.force:
        print(json.dumps({"status": "exists", "output_dir": str(output_dir)}, indent=2))
        return 0

    train_labels = np.load(args.train_labels, mmap_mode="r")
    val_labels = []
    with open(args.val_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val_labels.append(int(float(row["label"])))
    val_labels_arr = np.asarray(val_labels, dtype=np.int64)

    train_indices = stratified_indices(train_labels, args.train_size, args.seed)
    val_indices = stratified_indices(val_labels_arr, args.val_size, args.seed)

    train_rows = _read_train_rows(Path(args.train_row_metadata), set(train_indices.tolist()))
    val_rows = _read_val_rows(Path(args.val_csv), set(val_indices.tolist()))
    protein_vocab = build_vocab([row["protein_sequence"] for row in train_rows + val_rows])
    smiles_vocab = build_vocab([row["smiles"] for row in train_rows + val_rows])

    _write_tsv(train_tsv, train_rows)
    _write_tsv(val_tsv, val_rows)
    _dump_json(protein_vocab_json, {"stoi": protein_vocab})
    _dump_json(smiles_vocab_json, {"stoi": smiles_vocab})
    _dump_json(
        dataset_meta_json,
        {
            "dataset_name": "pubchem_course",
            "seed": args.seed,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "train_positive_rate": float(np.mean([int(row["label"]) for row in train_rows])),
            "val_positive_rate": float(np.mean([int(row["label"]) for row in val_rows])),
            "protein_vocab_size": len(protein_vocab),
            "smiles_vocab_size": len(smiles_vocab),
            "protein_pad_id": 0,
            "protein_unk_id": 1,
            "smiles_pad_id": 0,
            "smiles_unk_id": 1,
            "suggested_max_protein_len": 512,
            "suggested_max_smiles_len": 128,
            "source_train_row_metadata": args.train_row_metadata,
            "source_train_labels": args.train_labels,
            "source_val_csv": args.val_csv,
        },
    )

    summary = {
        "status": "built",
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "protein_vocab_size": len(protein_vocab),
        "smiles_vocab_size": len(smiles_vocab),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
