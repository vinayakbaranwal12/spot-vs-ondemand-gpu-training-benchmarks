# Harness

This directory holds the scripts that produced every number in `results/`: provisioning, training, checkpointing, simulated reclaims, resume timing, the capacity probe, and the tabulation.

## Dependencies

- `python3` 3.9 or later, with `matplotlib` for the charts (`requirements.txt` at the repo root pins the version that ran the analysis)
- `doctl`, authenticated, for Droplet and snapshot operations
- `s5cmd`, for Spaces uploads and downloads
- `docker`, on each GPU Droplet, for the training containers
- `ssh`/`scp`, key-based, for driving the Droplets

Training containers, by tag and digest:

- MI355X: `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0`
  (`rocm/pytorch@sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1`)
- B300: `nvcr.io/nvidia/pytorch:25.11-py3`
  (`nvcr.io/nvidia/pytorch@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21`)

Credentials are read from environment variables (`SPACES_KEY`, `SPACES_SECRET`, `SSH_KEY`) and pushed to the Droplets at run time. No credentials are stored in this repository.

## Script Inventory

| Script | Role |
| --- | --- |
| `train_bench.py` | The training and checkpointing workload: FSDP2, DCP sharded checkpoints, deterministic synthetic tokens, RNG-complete meta, event logging |
| `provision.sh` | Create or delete Droplets by ID (never by tag), record create timing |
| `setup_node.sh` | Mount the NVMe scratch disk, install `s5cmd`, prepare the run directories |
| `preflight.sh` | Environment gate: driver, container health, GPU visibility, disk and network snapshot |
| `measure_resume.sh` | One simulated reclaim: delete the Droplet, provision a replacement, time every resume phase |
| `measure_resume_existing.sh` | Same timing against an already-running replacement Droplet |
| `analyze.py` | Re-tabulates `checkpoint_save*.csv`, `resume.csv`, `overhead_model.csv`, and the charts from the raw events |
| `refresh_results_schema.py` | Schema migration helper for the results CSVs |
| `batch_watchdog.sh` | Deletes idle tagged Droplets when no training process is running; source of the idle-cost accounting |
| `probe_mi355x8_region.sh`, `probe_mi355x8_region_loop.sh` | The MI355X 8x capacity probe (one region sweep per pass) |
| `run_*.sh` | Batch drivers and one-off recovery scripts for the arms that produced the final rows |

Some recovery scripts carry `<droplet-id>`, `<droplet-ip>`, and `<snapshot-id>` placeholders where the original run referenced specific infrastructure. They are published for provenance; re-running them means supplying your own IDs.

## Workspace Layout

The scripts were executed inside a working workspace with this layout:

```text
workspace/
  logs/<gpu>/<model>/<run-id>/   events.jsonl, run_meta.json, resume_result.json
  results/                        CSVs and charts written by analyze.py
  scripts/                        this harness
```

In this published repository, the run workspace under `logs/` is published as `data/`, with the same `<gpu>/<model>/<run-id>` tree below it. The batch drivers reference the workspace layout they ran in; re-running the benchmark on your own infrastructure means recreating that workspace (or pointing `ROOT` at one).

## Tabulation

`analyze.py` takes an evidence root that contains the run tree under `logs/`, and writes CSVs to `<root>/results/` and charts to `<root>/reports/charts/`:

```bash
python3 harness/analyze.py --root <workspace-root>
```

Rows are excluded from the final artifacts through the `exclude_from_final_results` flag in each run's `run_meta.json`. The overhead model pools nothing: it combines the 8B checkpoint medians with only the matching resume arm (cold or pre-pulled snapshot) for each grid. See `METHODS.md` for the formulas.