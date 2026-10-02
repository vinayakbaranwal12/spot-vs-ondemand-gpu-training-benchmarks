# Methods

## Environment

| | MI355X 1x | B300 1x |
| --- | --- | --- |
| Size slug | `gpu-mi355x1-288gb-spot` | `gpu-b300x1-288gb-lc-spot` |
| Spot price (run date) | $2.97/hr | $8.00/hr |
| Region | mem1 | mkc1 |
| vCPU / RAM | 24 / 256 GiB | 28 / 448 GiB |
| Boot disk | 720 GB (`/dev/vda1`) | 720 GB (`/dev/vda1`) |
| Scratch | 5 TB NVMe (`/dev/vdc1`, mounted at `/scratch`) | 5 TB NVMe (`/dev/vdc1`, mounted at `/scratch`) |
| OS image | `gpu-amd-base` (Ubuntu 24.04, ROCm) | `gpu-h100x1-base` |
| Driver | ROCk module 6.19.14.31400000 | NVIDIA 580.173.02 (nvidia-smi CUDA 13.0) |
| Container | `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0` | `nvcr.io/nvidia/pytorch:25.11-py3` |
| Container digest | `sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1` | `sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21` |
| Image size | 46.30 GiB | 8.58 GiB |
| PyTorch | 2.12.0+rocm7.14.1 | 2.10.0a0+b558c986e8.nv25.11 |
| transformers | 5.18.0 (pip, installed at train/resume time) | 5.18.0 |
| Checkpoint bucket | `<bucket>` in atl1 (cross-region to mem1) | `<bucket>` in mkc1 (same region) |

Prices were re-queried on 2026-10-02T17:15:46Z and matched the run-date prices. Spot SKU only: no on-demand MI355X or B300 size existed in the size list at test time. Environment snapshots per GPU are in `environment/`.

## Design

- **Workload:** Qwen3 configurations at 1.7B and 8B parameters with random weights, FSDP2 (`fully_shard`) with fp32 master weights and AdamW state, bf16 compute, PyTorch Distributed Checkpoint (DCP) sharded checkpoints, synthetic deterministic token batches (seeded per step, so a resumed run sees the same batch at each step), sequence length 4096, batch size 4, grad accumulation 1.
- **Checkpoint arms:** three saves per clean run (steps 60, 80, 100), written to the local NVMe scratch disk and uploaded to Spaces. The 8B checkpoint is 91.54 GiB (12.00 bytes per parameter: fp32 master weight plus two fp32 AdamW moment tensors).
- **Reclaims were simulated** by deleting the Droplet mid-training and provisioning a replacement, in two variants: a **cold start** (fresh OS image, container pulled from the registry) and a **pre-pulled snapshot** (a Droplet snapshot that already contains the container image). Each resume downloads the checkpoint from Spaces, reinstalls the runtime pip package, loads the model, and runs to the first resumed optimizer step.
- **n=3 per arm**, medians reported. Phase times are per-run wall times; phase medians do not sum to the median of R.
- **R definition:** R = provision + env (image pull + container start) + download + pip install + load + first step, wall time from reclaim to the first resumed optimizer step. Capacity retry waits are tracked separately and are not part of R.

## Overhead Model

For a checkpoint interval I (seconds), checkpoint cost C, upload time U, resume time R, and a reclaim rate lambda (reclaims per hour, an assumption supplied by the reader):

```text
overhead = C / I + lambda * (I / 2 + U + R)
break_even_discount = overhead / (1 + overhead)
I_star = sqrt(2C / lambda)      (Young 1974, first-order optimum)
```

The break-even discount is the minimum spot discount at which spot is still cheaper than on-demand. Grid scope in `results/overhead_model.csv`: 8B checkpoint rows only, no pooling across model sizes or resume arms, one grid per GPU x resume arm (cold, pre-pulled snapshot) x upload mode (synchronous, asynchronous), at I in {10, 30, 60, 120} minutes and lambda in {0.01, 0.05, 0.1, 0.25}. Synchronous mode sets C = local save + upload and U = 0; asynchronous mode sets C = local save and U = upload. 1.7B rows appear in the checkpoint table but have no resume measurement, so they are not in the grid.

## Validity Checks

- **Strict resume correctness (B300, bit-exact):** a like-for-like deterministic run (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed 42) trained to step 100, saved ckpt-80, and resumed from it. Steps 81 to 100 matched the uninterrupted run bit for bit (`mismatch_count=0`). Artifact: `results/correctness/det_resume_bitexact_compare.json`.
- **Resume consistency:** all six B300 production resumes load the same ckpt-100 and report an identical step-101 loss, bit-exact across processes.
- **Old-format ablation:** stripping the newer RNG fields (`python_random_state`, `numpy_rng_bit_generator_state`, and the rest of the post-hoc additions) from `meta.json` still resumed bit-exactly on steps 81 to 100. The original checkpoint format was already complete for this harness; the added fields are hardening, and they add about 20 KB to a 91.5 GiB checkpoint. Artifact: `results/correctness/old_meta_ablation_compare.json`.
- **Nondeterminism scale (B300):** two independent from-scratch runs differ by 1.85e-4 in step-99 loss under default settings; a non-deterministic fresh-process continuation differs from the original run by 1.89e-3. The gap is kernel nondeterminism, not missing checkpoint state. Deterministic runs under identical settings match bit-exact. These deltas are recorded observations from the determinism verification runs, whose driver logs are not part of the published data (they carry local paths), so they cannot be re-derived from this repository alone.
- **Chronology of the strict-loss fields:** the `loss_match_strict` fields and notes inside the six published B300 `resume_result.json` files were written at resume time, before the bit-exact test above completed, so they read `NOT ESTABLISHED`. The establishing test is the published `det_resume_bitexact_compare.json`, which is why `METHODS.md` and the article state strict correctness as established. The MI355X final resume rows carry `loss_match_strict=True` at source.
- **Environment snapshot chronology:** the `environment/mi355x1/` snapshot was captured before the scratch mount fix, so it shows the NVMe scratch disk unmounted, which is the initial state that motivated the fix described under Deviations. The final runs ran with scratch mounted at `/scratch`.
- **MI355X final resume rows** carry `loss_match_strict=True` in `results/resume.csv`, verified against a deterministic scratch reference continuation.

## Exclusions

`results/run_index.csv` lists every run in the benchmark with its status and exclusion reason. The categories:

- **Smoke runs:** initial sanity checks (short steps, bf16 state); one unflagged B300 smoke checkpoint row is also excluded from the published `checkpoint_save_final.csv` and listed in the index.
- **Boot-disk runs:** clean checkpoint runs written before the scratch mount fix, plus their resumes; the boot-disk storage path itself sets the exclude flag in the tabulation.
- **DOCR overlap runs:** checkpoint runs whose uploads overlapped a registry push; excluded for contention.
- **Superseded or reconstructed resume rows:** rows whose provision time was reconstructed after a `create --wait` dropped, and early cold rows whose R omitted `pip_install_s` before it was accounted; replaced by timed replacements.
- **Data publication:** only the 16 final runs are published under `data/`. The excluded runs appear as index rows only.

## Deviations

- **Scratch mount fix:** the 5 TB NVMe scratch disk was not mounted by default on either image. The first MI355X checkpoints silently went to the boot disk, skewing save times and bloating a snapshot to 334.46 GiB (about $20.07/month at $0.06/GiB/month; recorded in `pricing/snapshot-mi355x1.md`). The disk was mounted explicitly and the clean runs were redone on scratch.
- **B300 container choice:** the newest NGC PyTorch container (26.07, CUDA 13.3.1) does not run natively on the driver in `gpu-h100x1-base` (580.173.02, CUDA 13.0). The benchmark used NGC 25.11 (CUDA 13.0.2), the newest release native on that driver, rather than benchmarking in CUDA forward-compatibility mode. The pre-run gate for every B300 run checked driver, container health, and a bf16 matmul; see `environment/b300x1/cuda_compat_check.json`.
- **DOCR variant not measured:** a registry-mirror container path was planned and its scripts exist, but those runs were excluded (upload contention) and the variant was never measured.
- **pip install inside R:** the `pip install transformers` step before `torchrun` was initially unaccounted wall time. It is included in R for every final row, which is why the early cold rows that omitted it were superseded.
- **Cross-region storage was forced:** creating a Spaces bucket in mem1 returned an InvalidRegion error, so the MI355X bucket lives in atl1 and its checkpoint downloads cross regions. The B300 bucket is in mkc1, co-located with its Droplet.

## Limitations

- B300 and MI355X do not share a software stack (PyTorch 2.10 on CUDA versus PyTorch 2.12 on ROCm). Cross-GPU differences in save, load, and other wall times may be partly software, so cross-GPU gaps are not a pure silicon comparison.
- Reclaims were simulated by Droplet deletion. Real reclaims add the notice window and provider-side teardown, which these timings do not include.
- No reclaim rate was measured; lambda in the overhead model is an assumption. Published spot prices were captured on 2026-10-02T17:15:46Z (`pricing/size_prices_capture.json`) and matched the run dates, but spot prices can change daily.
- The MI355X download phase spread (216 to 418 s across runs) exceeds the cold versus pre-pulled median R gap, so the 20 percent snapshot advantage is directional at n=3.
- Step-101 loss differs across independent runs by about 3e-3 under default (non-deterministic) settings, consistent with nondeterministic GPU kernels.
- Multi-node training, 8x sizes, real datasets, and an on-demand baseline for the same GPUs were not measured. The 8x result is a capacity finding: the MI355X 8x probe returned 422 on every create attempt (see `results/capacity_probe_mi355x8.csv`), and B300 8x was held.
- Spot GPU Droplets were in public preview during the run; capacity availability is a moving target and the 422 events in the final rows are part of the measured record.