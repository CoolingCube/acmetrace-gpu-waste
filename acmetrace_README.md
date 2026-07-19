# GPU Cluster Waste — AcmeTrace Analysis

**34–40% of GPU jobs fail. 49% of reserved GPU time is idle. Single-GPU jobs wait 2,538x longer than 512-GPU gang jobs.**

Three independent waste mechanisms, all from the same production cluster.
Real data from Shanghai AI Lab, published in NSDI 2024.

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

## Three waste mechanisms

### 1. Job failures

A failed job consumes GPU resources from submission until failure.
At 34–40% failure rates, a significant fraction of reserved GPU-hours
produce zero output.

| Cluster | GPU jobs | Failure rate | GPU-hrs wasted |
|---|---|---|---|
| Seren | 450,590 | 33.9% | 235,993 (4.2%) |
| Kalos | 19,907 | 39.7% | 93,153 (6.3%) |

69–73% of all GPU jobs finish in under 5 minutes.
Most of those short jobs are failures — GPU reserved, zero output.

### 2. Within-job non-GPU phases

From the NSDI 2024 paper on these exact clusters:
only 51% of reserved GPU time is used for actual GPU computation.
The remaining 49% is model weight loading, CPU preprocessing,
correctness testing, and orchestration.

```
Total structural waste = within-job idle (49%) + failure waste (4–6%)
                       = 53–55%
```

### 3. Queue starvation — the counterintuitive finding

Large gang jobs schedule faster than stuck single-GPU jobs.

| Cluster | Job size | Avg queue | Long wait (>1hr) | Failure rate |
|---|---|---|---|---|
| Seren | 1 GPU | 924s (21.7h stuck) | 1.1% | 34.3% |
| Seren | 512+ GPU | **31s** | **0.0%** | 4.7% |
| Kalos | 1 GPU | 185s (1.6h stuck) | 0.8% | 42.3% |
| Kalos | 512+ GPU | **64s** | **0.5%** | 9.5% |

**512+ GPU gang jobs schedule 2,538x faster than stuck single-GPU eval jobs.**

This is not a gang scheduling problem. It is a queue fairness problem.
Large pretraining jobs have dedicated quotas and high priority.
292,000 single-GPU eval jobs compete in crowded low-priority queues.
1.1% get starved and wait 20+ hours.

GPU-hours wasted waiting (Seren): 98,989 GPU-hours over 6 months.

---

## Cluster results

### Seren cluster — 818,327 jobs, 6 months

| Metric | Value |
|---|---|
| GPU jobs | 450,590 |
| Completed | 46.1% |
| Failed | 33.9% |
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
| Failed | 39.7% |
| Cancelled | 6.2% |
| Total GPU-hours reserved | 1,481,439 |
| GPU-hours wasted (failures) | 93,153 (6.3%) |
| Jobs under 5 minutes | 73% |

---

## Run the analysis

```bash
pip install pandas numpy matplotlib
```

**Job failure and within-job waste:**
```bash
python acmetrace_waste_analysis.py
```

**Queue starvation analysis:**
```bash
python acmetrace_queue_starvation.py
```

No login required. Downloads directly from GitHub (CC-BY-4.0).

---

## Relation to other waste analyses

AcmeTrace measures cluster-level job waste (training + evaluation).
The [structural-gpu-waste](https://github.com/CoolingCube/structural-gpu-waste)
repo measures inference-time request burstiness waste.

Both are structural. Both are independent. A production cluster has both:

| Layer | Mechanism | Measured by |
|---|---|---|
| Between requests | Burstiness | Azure, BurstGPT, ServeGen |
| Within jobs | Failures + non-GPU phases | AcmeTrace (script 1) |
| Queue fairness | Starvation of small jobs | AcmeTrace (script 2) |

---

## Data

- **AcmeTrace**: [InternLM/AcmeTrace](https://github.com/InternLM/AcmeTrace) (CC-BY-4.0)
- **Paper**: Hu et al., "Characterization of Large Language Model Development
  in the Datacenter", NSDI 2024

*Code: MIT License*
