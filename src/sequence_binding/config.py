from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


@dataclass
class ExperimentSection:
    name: str
    seed: int
    output_dir: str


@dataclass
class DataSection:
    train_tsv: str
    val_tsv: str
    protein_vocab_json: str
    smiles_vocab_json: str
    max_protein_len: int
    max_smiles_len: int
    num_workers: int
    pin_memory: bool


@dataclass
class ModelSection:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_multiplier: int
    dropout: float
    mamba_d_state: int
    mamba_d_conv: int
    mamba_expand: int


@dataclass
class TrainSection:
    epochs: int
    global_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    mixed_precision: str
    log_interval: int
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


@dataclass
class DistributedSection:
    strategy: str
    backend: str
    find_unused_parameters: bool


@dataclass
class ReportSection:
    report_dir: str


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection
    data: DataSection
    model: ModelSection
    train: TrainSection
    distributed: DistributedSection
    report: ReportSection

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_section(cls, payload: dict[str, Any]):
    return cls(**payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig(
        experiment=_build_section(ExperimentSection, raw["experiment"]),
        data=_build_section(DataSection, raw["data"]),
        model=_build_section(ModelSection, raw["model"]),
        train=_build_section(TrainSection, raw["train"]),
        distributed=_build_section(DistributedSection, raw["distributed"]),
        report=_build_section(ReportSection, raw["report"]),
    )
