# Spot vs On-Demand GPU Training Benchmarks

A benchmark that measures what an interruption actually costs during model training on spot GPUs, so the spot versus on-demand decision can be made from measured numbers. I trained 1.7B and 8B parameter models with FSDP2 and sharded PyTorch Distributed Checkpoint state on two spot GPU Droplets (AMD Instinct MI355X 1x and NVIDIA B300 1x), simulated reclaims by deleting each Droplet mid-training and provisioning a replacement, and timed every phase of saving, uploading, and resuming. This repository publishes the harness, the raw run logs for the final runs, and the tabulated results, so every number can be independently re-tabulated from the raw data.

DigitalOcean Community article: link added on publication.

## Headline Results

Checkpoint cost (save plus upload, median of n=3 per cell) and resume time R (median of n=3, wall time from reclaim to the first resumed optimizer step) for the 8B checkpoint (91.54 GiB, 98.3 GB, 12.00 bytes per parameter).

| GPU | Local save | Upload | R, cold start | R, pre-pulled snapshot |
| --- | ---: | ---: | ---: | ---: |
| MI355X 1x | 50.2 s | 156.8 s | 706.5 s | 564.9 s |
| B300 1x | 134.2 s | 113.2 s | 436.6 s | 490.5 s |

The pre-pulled snapshot finding, stated plainly: the snapshot cut MI355X's resume time by 20 percent and added 12 percent on B300, because a snapshot boots slower (+45 s on MI355X, +145 s on B300) and pays off only when the environment phase it skips (image pull plus container start: 188 s for the 46.30 GiB MI355X image, 96 s for the 8.58 GiB B300 image) takes longer than that penalty. The full phase breakdown, the break-even discount grids, and the capacity probe are in `results/`.

## Repository Layout

```text
spot-vs-ondemand-gpu-training-benchmarks/
  README.md
  METHODS.md
  LICENSE                       (MIT)
  CITATION.cff
  requirements.txt              (analysis only)
  SHA256SUMS.txt
  harness/                      train_bench.py, analyze.py, provision, resume, probe, batch scripts, harness README
  data/<gpu>/<model>/<run-id>/  events.jsonl, run_meta.json, resume_result.json (final runs only)
  environment/<gpu>/            versions, GPU info, disk and network, container and timing snapshots
  results/                      checkpoint_save_final.csv, resume.csv, overhead_model.csv,
                                capacity_probe_mi355x8.csv, run_index.csv, charts/, correctness/
  pricing/                      spot size prices with capture timestamps, snapshot sizes and costs
```

`data/` holds the 16 final runs. Every run in the whole benchmark, including the excluded ones with their exclusion reasons, is listed in `results/run_index.csv`.

## Re-Tabulate From The Raw Data

`harness/analyze.py` recomputes the checkpoint, resume, and overhead CSVs directly from the raw event logs, never from a cached summary. It expects the run tree under `logs/` inside its root, so this copies `data/` into a scratch root first, which leaves the checksummed tree untouched:

```bash
pip install -r requirements.txt
mkdir -p /tmp/spot-repro && ln -s "$PWD/data" /tmp/spot-repro/logs
python3 harness/analyze.py --root /tmp/spot-repro
diff /tmp/spot-repro/results/checkpoint_save_final.csv results/checkpoint_save_final.csv && echo "matches"
diff /tmp/spot-repro/results/overhead_model.csv results/overhead_model.csv && echo "matches"
```

Expected result: both diffs are empty. This was verified before this repository was committed. The regenerated `resume.csv` contains the final rows; the committed `resume.csv` additionally keeps the excluded rows (flagged in the `exclude_from_final` column) for transparency, whose raw data is intentionally not published. One internal checkpoint row from an unflagged B300 smoke run is not part of the published `checkpoint_save_final.csv`; it is recorded as excluded in `results/run_index.csv`.

## Re-Run On Your Own GPU Droplet

Required tools: `doctl` (authenticated), `ssh`/`scp` with a key pair, `docker` on the Droplet, `s5cmd`, and Python 3.9 or later. See `harness/README.md` for the container tags and digests and the environment variables the scripts read.

```bash
export SSH_KEY=~/.ssh/id_ed25519          # key registered with DigitalOcean
export SPACES_KEY=... SPACES_SECRET=...  # Spaces credentials for checkpoint upload
./harness/provision.sh create <size-slug> <region>
./harness/setup_node.sh <ip>
./harness/run_full_mi355x1.sh            # or run_full_b300x1.sh for the NVIDIA arm
./harness/measure_resume.sh               # one simulated reclaim, timed per phase
python3 harness/analyze.py --root <workspace-root>
```

Sizes used in this run: `gpu-mi355x1-288gb-spot` (mem1) and `gpu-b300x1-288gb-lc-spot` (mkc1). Mount the NVMe scratch disk before the first checkpoint: it was not mounted by default on these images, and checkpoints written to the boot disk skew the timings (see `METHODS.md`, Deviations).

## Integrity

Every file in this repository, except `SHA256SUMS.txt` and `.gitignore` themselves, is checksummed:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

## Citation

See `CITATION.cff`. A plain-text form:

Baranwal, Vinayak. "Spot vs on-demand GPU training benchmarks: checkpoint, reclaim, and resume cost on MI355X and B300." 2026. `https://github.com/vinayakbaranwal12/spot-vs-ondemand-gpu-training-benchmarks`

## License

MIT. See `LICENSE`.