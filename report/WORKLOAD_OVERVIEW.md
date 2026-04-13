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
