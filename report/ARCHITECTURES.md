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
