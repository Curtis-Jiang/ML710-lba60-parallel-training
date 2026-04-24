# Architectures

## Attention

The attention model uses:

- learned token embeddings
- learned positional embeddings
- a small transformer encoder on the protein branch
- a small transformer encoder on the SMILES branch
- masked mean pooling
- a fusion MLP over `[protein, smiles, |diff|, product]`

Current course-scale parameter count:

- `15,651,841`

## Mamba

The mamba model keeps the same outer scaffold but replaces each branch encoder
with state-space sequence blocks.

Current course-scale parameter count:

- `18,926,593`

The final reported results use `mamba_ssm`.

---

# v2 Architecture Notes – Parallelism Entry Points

The architectures themselves are unchanged between v1 and v2 (same
parameter counts, same fusion head). v2 only adds where the model is sharded
or split for each parallel strategy so the systems story in
`COURSE_EXPERIMENT_SUMMARY.md` (v2) is unambiguous.

## v2 Attention – Where Parallelism Hooks In

- `single`: full model on one GPU.
- `ddp`: whole model replicated per rank; all-reduce on gradients.
- `ddp_zero` (ZeRO-1): optimizer states sharded across ranks; params and grads
  still replicated.
- `fsdp_z2`: optimizer states + gradients sharded (full shard except params).
- `fsdp_z3`: full sharding — parameters, gradients, and optimizer states are
  all sharded; params are gathered on-demand per forward/backward.
- `tp` (Megatron-style): each encoder attention block is split column-then-row
  across ranks, with 2 `all-reduce` calls per block (4 per fwd + bwd pass).
  Same weights, same init, only the layout across GPUs differs.
- `branch_mp`: the two branch encoders (protein / SMILES) live on different
  GPUs; the fusion MLP is replicated. Communication is limited to forwarding
  the two branch embeddings into the fusion step once per batch.
- `hybrid_tp2_dp2`: 4-GPU 2D mesh of `TP=2 x DP=2`; each TP-pair runs the
  Megatron-style split above, and the two TP-pairs form a DP group.

## v2 Mamba – Parallelism Coverage

In v2, Mamba is exercised with:

- `single` (`mamba_single_course`)
- `ddp` (`mamba_ddp_course`, 2-GPU, reports 1.75x speedup and 87% scaling
  efficiency over `mamba_single_course`)

Other strategies (FSDP, TP, branch_mp, hybrid) were validated via smoke tests
but not scheduled as full course-scale Mamba runs, because the systems
comparison is most directly demonstrated on the lighter Attention model.

## v2 Shared Components

These pieces are identical across all strategies and do not move between GPUs
differently per strategy:

- learned token + positional embeddings
- masked-mean pooling per branch
- fusion MLP over `[protein, smiles, |diff|, product]`
- binary classification head

This keeps the v2 throughput and quality differences attributable to the
sharding / communication pattern of each strategy, not to model-side changes.
