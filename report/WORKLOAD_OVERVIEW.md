# Workload Overview

The project workload is binary binding prediction between:

- one protein sequence
- one ligand SMILES string

The design goal is to keep the workload simple enough that distributed systems
questions stay in focus:

- the inputs are simple strings, tokenized at runtime
- the labels are binary and fully supervised
- the same data pipeline feeds both model families

The compact dataset uses:

- `100,000` train rows
- `10,000` val rows
- character-level protein tokenization
- character-level SMILES tokenization

This keeps the repo better aligned with the ML710 goal of studying parallel
training behavior instead of domain-specific preprocessing complexity.

In the current final project version:

- Attention is the lighter comparison model
- Mamba with `mamba_ssm` is the heavier comparison model
- `ddp_zero` is the advanced distributed strategy

This gives one workload with two model families and one advanced distributed
method, which is a much cleaner course story than the original research-style
3D graph pipeline.

---

# v2 Workload Notes – Expanded Parallel-Strategy Study

The v1 framing above is kept: the dataset, tokenization, label definition, and
model families are unchanged. The v2 sweep reuses the **same workload** as the
controlled variable, and varies only the parallelism strategy and GPU count so
that all comparisons in `COURSE_EXPERIMENT_SUMMARY.md` (v2) are apples-to-apples.

## v2 What Is Held Constant

- dataset: `100,000` train / `10,000` val rows (compact PubChem course split)
- task: binary protein-ligand binding
- truncation: protein `512`, SMILES `128`
- tokenization: character-level for both branches
- optimizer: AdamW, `lr=1e-3`, `weight_decay=0.01`
- epochs: `25`
- global batch size: `64`
- mixed precision: `bf16`
- hardware: NVIDIA H100 80GB (single node, up to 4 GPUs)

## v2 What Varies

- model family: `attention` or `mamba (mamba_ssm)`
- parallel strategy: `single`, `ddp`, `ddp_zero (ZeRO-1)`, `fsdp_z2`, `fsdp_z3`,
  `tp`, `branch_mp`, `hybrid_tp_dp`
- GPU count: `1`, `2`, or `4`

Because the workload is fixed, differences in throughput, peak memory, and
validation quality across runs are attributable to the strategy and scale,
which is the intended systems comparison for the report.

## v2 Role of Each Strategy

- `single`: quality/throughput anchor per model family
- `ddp`: systems-level data parallel reference
- `ddp_zero` / `fsdp_z2` / `fsdp_z3`: progressively more aggressive memory
  sharding (optimizer → gradients → parameters)
- `tp`: Megatron-style column/row-split attention, stress test for collective
  overhead at small hidden dim
- `branch_mp`: task-aware model parallelism across the protein / SMILES branches
- `hybrid_tp_dp`: 2D parallelism (TP2 x DP2) as the highest-dimensional strategy
  in the matrix
