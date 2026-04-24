# Slide Outline

## v1 Outline (original 4-run story)

1. Problem statement and course framing
2. Dataset and compact TSV representation
3. Data pipeline and character tokenization
4. Attention architecture
5. Mamba architecture and `mamba_ssm` backend
6. Single-GPU baselines
7. 2-GPU DDP baseline
8. 2-GPU DDP + ZeRO strategy
9. Final four-run results table
10. Throughput comparison
11. Validation quality comparison
12. Lessons learned and next steps

---

## v2 Outline – Full Parallel-Strategy Sweep

This deck structure is what the Phase B / C / D sweep now supports. Numbers
and talking points are sourced from `report/COURSE_EXPERIMENT_SUMMARY.md`
(v2 section) and `report/PHASE_COMPLETION_REPORT.md`.

1. Problem statement and course framing
2. Dataset and compact TSV representation
3. Data pipeline and character tokenization
4. Attention and Mamba architectures (`mamba_ssm` backend)
5. Experiment matrix overview – Phase B / C / D (see `EXPERIMENT_MATRIX.md` v2)
6. Phase B – single-GPU baselines
   - attention 1292 ex/s, AUC 0.9443
   - mamba 788 ex/s, AUC 0.9568
7. Phase C Round 1 – DDP vs DDP+ZeRO-1 (2 GPU)
   - throughput: DDP 1980 vs ZeRO-1 1845 ex/s
   - peak memory: ZeRO-1 slightly lower than DDP
8. Phase C Round 2 – FSDP ZeRO-2 vs ZeRO-3 (2 GPU)
   - throughput vs memory trade-off across shard levels
9. Phase C Round 3 – Tensor Parallelism and Branch Model Parallelism (2 GPU)
   - why TP underperforms at this model scale (collective overhead)
   - branch_mp preserves AUC (0.9430) with modest throughput gain
10. Phase D – 4-GPU serial sweep
    - best throughput: DDP 2433 ex/s (1.88x, 47% scaling efficiency)
    - best memory: FSDP ZeRO-3 0.82 GB peak (-75% vs single)
    - hybrid TP2xDP2 4-GPU as the 2D parallelism case study
11. Scaling-efficiency summary + how to read "Eff vs 1/N" correctly
12. Throughput-vs-quality Pareto frontier
13. Mamba vs Attention quality/throughput trade-off (with `mamba_ddp_course` filled in once available)
14. Lessons learned
    - DDP family scales monotonically in memory, nearly monotonically in throughput
    - TP / hybrid are comms-bound for ~16 M-param models on H100
    - ZeRO-3 wins when model size pressure appears; DDP wins when it doesn't
15. Next steps
    - Pipeline parallelism, 3D parallelism, and multi-node extension
