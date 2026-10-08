### Jaehoon Yang

M.S. student at Seoul National University, Architecture and Code Optimization (ARC) Lab,
advised by Prof. Jae W. Lee. I work on **LLM inference runtimes** — KV-cache memory management,
MoE execution, and GPU memory virtualization. My research ships as patches to production serving
engines rather than as simulators.

---

**[MOLT](https://arxiv.org/abs/2610.05748)** — fine-grained GPU memory sharing for LLM serving

Lets inference reclaim memory from a *running* LoRA fine-tuning step at the granularity of individual
saved activations; the step survives and recomputes them in its backward pass. Slots are backed and
unbacked through CUDA virtual memory management, and stay correct under CPU–GPU dispatch asynchrony
and tensor parallelism. Implemented as a **2.7K-line patch across 23 files of vLLM v0.11.0**, plus a
9.7K-line module.

> ≥99.7% inference SLO attainment while tuning runs, at **1.9–3.3× the tuning throughput** of
> discard-based sharing. Four deployments, 24B–70B, on H100 SXM and B200.

**[Libra](https://github.com/SNU-ARC/Libra)** — load balancing for large-scale MoE inference · **ICLR 2026**

Predicts upcoming expert activations and hides the balancing cost by restructuring execution into
locality-aware local/remote phases, with AllGather dispatch so token sharding overlaps compute.
Expert-replication planning and token sharding are written in **Cython** and compiled as native
modules in SGLang v0.4.10's build.

> **Up to 19.2% throughput** on two state-of-the-art MoE models across 8×H200.

**[Lachesis](https://arxiv.org/abs/2610.08378)** — lifetime-aware KV cache placement across HBM and high-bandwidth flash

A placement layer between the agent harness and the serving engine, exposed to the engine as two
calls. Agent KV lifetime is set by the harness, so context segments are classified along temporal,
structural and inter-worker axes, and the KV allocator keeps per-tier free lists behind a block-hash
prefix cache.

> **1.19–3.13× flash endurance** — 3.3–12.2 device-years, against 1.9–4.8 for HBM-first placement.

---

[blueblack319.github.io](https://blueblack319.github.io/) ·
[Google Scholar](https://scholar.google.com/citations?user=D2uagGEAAAAJ)
