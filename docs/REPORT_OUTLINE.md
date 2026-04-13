# Report Outline

1. Problem and workload
   Present `lba60` as a single protein-ligand affinity regression workload and explain why the repo was simplified to one task.

2. Baseline system
   Describe the single-GPU training path, data size, default config, and runtime artifacts.

3. Parallel strategies
   Compare single GPU, 2-GPU DDP, and packed concurrent jobs under a fixed GPU budget.

4. Metrics
   Report throughput, wall-clock time, validation Pearson, test Pearson/RMSE, and memory usage.

5. Statistical efficiency
   Discuss whether faster configurations also preserve or degrade convergence quality.

6. Goodput
   Under a fixed GPU budget, compare one wider run versus multiple concurrent runs.

7. Final recommendation
   State which configuration is best for the course constraints and why.

8. Limitations
   Be explicit about what was intentionally removed from the original research project: MoE, expert stacking, and the extra datasets.
