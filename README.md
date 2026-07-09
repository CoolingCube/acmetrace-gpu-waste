# GPU Cluster Waste — AcmeTrace Analysis

**34–40% of GPU jobs fail. 49% of reserved GPU time is idle inside running jobs.**

This repository analyzes structural GPU waste in LLM training and inference clusters
using AcmeTrace — real production data from Shanghai AI Lab, published in NSDI 2024.

---

## What AcmeTrace is

AcmeTrace is a job trace dataset from two real GPU clusters at Shanghai AI Lab
(the team behind InternLM), covering March–August 2023.

- 880,740 total jobs across two clusters
- 470,497 GPU jobs
- Job types: Pretraining, SFT fine-tuning, Evaluation, MLLM, Debug
- Metrics: job state, duration, queue time, GPU count, GPU-hours consumed

Published in: *Characterization of Large Language Model Development in the Datacenter*
(Hu et al., NSDI 2024)

---

## Results

### Seren cluster — 818,327 jobs, 6 months

| Metric | Value |
|---|---|
| GPU jobs | 450,590 |
| Completed | 46.1% |
| **Failed** | **33.9%** |
| Cancelled | 10.2% |
| Preempted | 9.7% |
| Total GPU-hours reserved | 5,626,520 |
| GPU-hours wasted (failures) | 235,993 (4.2%) |
| Jobs under 5 minutes | 69% |

### Kalos cluster — 62,413 jobs, 6 months

| Metric | Value |
|---|---|
| GPU jobs | 19,907 |
| Completed | 54.1% |
| **Failed** | **39.7%** |
| Cancelled | 6.2% |
| Total GPU-hours reserved | 1,481,439 |
| GPU-hours wasted (failures) | 93,153 (6.3%) |
| Jobs under 5 minutes | 73% |

---

## Two sources of within-job waste

This is different from inference request burstiness (see
[structural-gpu-waste](https://github.com/CoolingCube/structural-gpu-waste)
for that analysis). AcmeTrace shows waste **inside** running jobs.

**1. Job failures**
A failed job consumes GPU resources from submission until failure.
At 34–40% failure rates, a significant fraction of reserved GPU-hours
produce zero output.

**2. Non-GPU phases inside jobs**
From the NSDI 2024 paper on these exact clusters:
only 51% of reserved GPU time is used for actual GPU computation.
The remaining 49% is:
- Model weight loading (~15% of job time)
- CPU preprocessing and tokenization
- Correctness testing (evaluation jobs: 19% of time)
- Orchestration and I/O

```
Total structural waste = within-job idle (49%) + failure waste (4–6%)
                       = 53–55%
```

69–73% of all GPU jobs finish in under 5 minutes.
Most of those short jobs are failures — GPU reserved, no output.

---

## Why this happens

Large model jobs fail often because:
- OOM errors when model size exceeds available VRAM
- NCCL communication failures in multi-node jobs
- Node failures and preemptions
- Configuration errors caught only at runtime

And jobs are idle inside their reservation because:
- Loading a 70B model takes minutes regardless of actual inference time
- Evaluation jobs run CPU-side correctness checks after GPU inference
- Tokenization and data preprocessing run on CPU before GPU is used

These are not scheduling problems. They are structural — the job boundary
includes phases where the GPU cannot be used regardless of how well
the scheduler is tuned.

---

## Run the analysis

```bash
pip install pandas numpy matplotlib
python acmetrace_waste_analysis.py
```

No login required. Downloads directly from GitHub (CC-BY-4.0).

---

## Relation to inference-time waste

AcmeTrace measures cluster-level job waste (training + evaluation).
The [structural-gpu-waste](https://github.com/CoolingCube/structural-gpu-waste)
repo measures inference-time request burstiness waste.

Both are structural. Both are independent. A production cluster has both:

| Layer | Mechanism | Measured by |
|---|---|---|
| Between requests | Burstiness | Azure, BurstGPT, ServeGen |
| Within jobs | Failures + non-GPU phases | AcmeTrace |

---

## Data

- **AcmeTrace**: [InternLM/AcmeTrace](https://github.com/InternLM/AcmeTrace) (CC-BY-4.0)
- **Paper**: Hu et al., "Characterization of Large Language Model Development
  in the Datacenter", NSDI 2024

*Code: MIT License*
