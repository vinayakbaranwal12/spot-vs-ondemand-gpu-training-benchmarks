#!/usr/bin/env python3
"""Rewrite results CSVs to the unified schema and refresh manifest runs[]."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(".")

CKPT_FIELDS = [
    "gpu", "model", "step", "param_count", "ckpt_bytes", "bytes_per_param", "master_dtype",
    "batch_size", "tokens_per_step", "local_save_s", "upload_s", "upload_gbps", "step_time_s",
    "overlap_docr_push", "exclude_from_final", "exclude_reason", "run_id", "timestamp",
]

RESUME_FIELDS = [
    "gpu", "model", "variant", "provision_s", "env_s", "image_pull_s", "container_start_s",
    "image_size_bytes", "download_s", "load_s", "first_step_s", "R_s", "steps_lost",
    "step_time_s", "batch_size", "tokens_per_step", "capacity_retries", "retry_wait_s",
    "loss_match", "run_id", "exclude_from_final", "precision_label",
]


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    # --- checkpoint_save from overlap JSON + placeholders for pending clean ---
    overlap = json.loads((ROOT / "results/checkpoint_docr_overlap.json").read_text())
    best = {}
    for r in overlap:
        key = (r["model"], r["step"])
        if key not in best or float(r.get("upload_s") or 0) > float(best[key].get("upload_s") or 0):
            best[key] = r
    ckpt_rows = []
    for (model, step), r in sorted(best.items()):
        ckpt_bytes = r.get("ckpt_bytes") or 0
        upload_s = r.get("upload_s") or 0
        upload_gbps = ""
        if ckpt_bytes and upload_s:
            upload_gbps = (float(ckpt_bytes) * 8 / float(upload_s)) / 1e9
        ckpt_rows.append({
            "gpu": "MI355X-1x",
            "model": model,
            "step": step,
            "param_count": 1720574976 if model == "1.7B" else 8190735360,
            "ckpt_bytes": ckpt_bytes,
            "bytes_per_param": r.get("bytes_per_param"),
            "master_dtype": "fp32",
            "batch_size": 1,
            "tokens_per_step": 4096,
            "local_save_s": r.get("local_save_s"),
            "upload_s": upload_s,
            "upload_gbps": upload_gbps,
            "step_time_s": "",
            "overlap_docr_push": True,
            "exclude_from_final": True,
            "exclude_reason": "upload overlapped DOCR push mem1->mkc1 (push started 2026-09-30T20:35:48Z)",
            "run_id": f"full-mi355x1-{model}-overlapped",
            "timestamp": r.get("upload_end_utc"),
        })
    write_csv(ROOT / "results/checkpoint_save.csv", CKPT_FIELDS, ckpt_rows)
    # final is empty until clean replacements
    write_csv(ROOT / "results/checkpoint_save_final.csv", CKPT_FIELDS, [])

    # --- resume.csv smoke row with full schema ---
    smoke_resume = json.loads(
        (ROOT / "logs/mi355x1/1.7B/smoke-mi355x1-1p7b-20260930195024/resume/resume_result.json").read_text()
    )
    # env_timing not split on smoke; image size from later full run snapshot if available
    img_bytes = ""
    snap = ROOT / "snapshots/mi355x1"
    for p in snap.rglob("env_timing.json"):
        et = json.loads(p.read_text())
        img_bytes = et.get("image_size_bytes", "")
        break
    resume_rows = [{
        "gpu": "MI355X-1x",
        "model": "1.7B",
        "variant": "cold",
        "provision_s": smoke_resume.get("provision_s"),
        "env_s": smoke_resume.get("env_s"),
        "image_pull_s": "",  # not split on smoke
        "container_start_s": "",
        "image_size_bytes": img_bytes,
        "download_s": smoke_resume.get("download_s"),
        "load_s": smoke_resume.get("load_s"),
        "first_step_s": smoke_resume.get("first_step_s"),
        "R_s": smoke_resume.get("R_s"),
        "steps_lost": "",
        "step_time_s": "",
        "batch_size": 1,
        "tokens_per_step": 4096,
        "capacity_retries": smoke_resume.get("capacity_retries", 0),
        "retry_wait_s": smoke_resume.get("retry_wait_s", 0),
        "loss_match": smoke_resume.get("loss_match"),
        "run_id": "smoke-mi355x1-1p7b-20260930195024",
        "exclude_from_final": True,
        "precision_label": "smoke, bf16 state",
    }]
    write_csv(ROOT / "results/resume.csv", RESUME_FIELDS, resume_rows)

    # --- manifest runs ---
    man_path = ROOT / "manifest.json"
    m = json.loads(man_path.read_text())
    m["runs"] = [
        {
            "run_id": "smoke-mi355x1-1p7b-20260930195024",
            "phase": "smoke_train_upload",
            "status": "completed",
            "droplet_id": "<droplet-id>",
            "recovered_run": True,
            "precision_label": "smoke, bf16 state",
            "exclude_from_final": True,
        },
        {
            "run_id": "smoke-mi355x1-1p7b-20260930195024-resume",
            "phase": "smoke_simulated_reclaim_resume",
            "status": "completed",
            "droplet_id": "<droplet-id>",
            "recovered_run": False,
            "precision_label": "smoke, bf16 state",
            "exclude_from_final": True,
            "R_s": smoke_resume.get("R_s"),
        },
        {
            "run_id": "full-mi355x1-1.7B-20260930203428",
            "phase": "full_ckpt_1.7B",
            "status": "failed_s5cmd_verify_EXIT_1",
            "droplet_id": "<droplet-id>",
            "recovered_run": True,
            "note": "first ckpt saved; upload verify used bare s5cmd; host-uploaded ckpt-60",
            "exclude_from_final": True,
            "exclude_reason": "overlap_docr_push and incomplete ckpt set",
        },
        {
            "run_id": "full-mi355x1-1.7B-continue",
            "phase": "continue_1.7B_then_8B",
            "status": "partial_EXIT_1_s5cmd_verify",
            "droplet_id": "<droplet-id>",
            "recovered_run": True,
            "note": "1.7B ckpt-80 uploaded; 8B ckpt-60 uploaded; both overlap DOCR push",
            "exclude_from_final": True,
            "exclude_reason": "overlap_docr_push",
        },
        {
            "run_id": "full-mi355x1-finish-aborted",
            "phase": "finish_remaining_ckpts",
            "status": "aborted",
            "droplet_id": "<droplet-id>",
            "recovered_run": False,
            "note": "aborted after DOCR overlap analysis to avoid more contaminated uploads",
            "exclude_from_final": True,
        },
        {
            "run_id": "clean-mi355x1-pending",
            "phase": "clean_replacement_ckpts",
            "status": "waiting_docr_push",
            "droplet_id": "<droplet-id>",
            "recovered_run": False,
            "note": "auto batch_size targeting ~85% VRAM; 3 ckpts per model after push_ok + preflight",
        },
    ]
    m.pop("notes", None)
    m["notes"] = [
        "DOCR push of spot-bench-rocm-pytorch started 2026-09-30T20:35:48Z; overlapping uploads excluded.",
        "Clean replacement checkpoints use auto batch_size (~85% VRAM) and run only after push completes.",
    ]
    man_path.write_text(json.dumps(m, indent=2) + "\n")
    print("manifest runs:", len(m["runs"]))
    print("checkpoint_save rows:", len(ckpt_rows))
    print("resume rows:", len(resume_rows))


if __name__ == "__main__":
    main()
