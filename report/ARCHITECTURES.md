# Architectures

## Attention

The attention model uses:

- learned token embeddings
- learned positional embeddings
- a small transformer encoder on the protein branch
- a small transformer encoder on the SMILES branch
- masked mean pooling
- a fusion MLP over `[protein, smiles, |diff|, product]`

## Mamba

The mamba model keeps the same outer scaffold but replaces each branch encoder
with state-space sequence blocks.

When `mamba-ssm` is available, the branch uses official Mamba blocks. If the
package is not installed, the repo falls back to a lightweight gated sequence
block so smoke validation still works.
