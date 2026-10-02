#!/usr/bin/env python3
"""Parse events.jsonl -> CSVs -> overhead model -> charts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_events(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in root.rglob("events.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            e["_path"] = str(p)
            rows.append(e)
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def median_or_none(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.median(xs)


def build_checkpoint_rows(events: List[Dict[str, Any]], meta_by_run: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # group by events file
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_file[e["_path"]].append(e)
    rows = []
    for path, evs in by_file.items():
        meta = meta_by_run.get(path, {})
        saves = [e for e in evs if e.get("event") == "ckpt_save_end"]
        uploads = [e for e in evs if e.get("event") == "upload_end"]
        steps = [e for e in evs if e.get("event") == "step"]
        step_times = [e["step_time_s"] for e in steps if "step_time_s" in e]
        step_time = median_or_none(step_times) if step_times else ""
        for i, s in enumerate(saves):
            up = uploads[i] if i < len(uploads) else {}
            ckpt_bytes = s.get("ckpt_bytes") or up.get("ckpt_bytes") or ""
            upload_s = up.get("duration_s", "")
            upload_gbps = ""
            if ckpt_bytes and upload_s:
                try:
                    upload_gbps = (float(ckpt_bytes) * 8 / float(upload_s)) / 1e9
                except Exception:
                    upload_gbps = ""
            overlap = s.get("overlap_docr_push", up.get("overlap_docr_push", meta.get("overlap_docr_push", "")))
            exclude = meta.get("exclude_from_final_results", False)
            exclude_reason = meta.get("exclude_reason", "")
            if overlap is True or str(overlap).lower() == "true":
                exclude = True
                exclude_reason = exclude_reason or "overlap_docr_push"
            storage = meta.get("storage", "")
            if not storage:
                # Historical clean/full runs wrote to /mnt on boot disk before DOSCRATCH mount
                if str(meta.get("run_id", "")).startswith(("clean-mi355x1-", "full-mi355x1-")):
                    storage = "boot_disk"
                elif str(meta.get("run_id", "")).startswith("clean-scratch-"):
                    storage = "scratch"
            if storage == "boot_disk" and not exclude:
                exclude = True
                exclude_reason = exclude_reason or "storage_boot_disk"
            # First durable save in a run is slower (allocator / page-cache / DCP warm-up)
            first_checkpoint = (i == 0)
            rows.append({
                "gpu": meta.get("gpu", ""),
                "model": meta.get("model", ""),
                "step": s.get("step", ""),
                "param_count": s.get("param_count", meta.get("param_count", "")),
                "ckpt_bytes": ckpt_bytes,
                "bytes_per_param": s.get("bytes_per_param", ""),
                "master_dtype": s.get("master_dtype", meta.get("master_dtype", "")),
                "batch_size": s.get("batch_size", meta.get("batch_size", "")),
                "tokens_per_step": s.get("tokens_per_step", meta.get("tokens_per_step", "")),
                "local_save_s": s.get("duration_s", ""),
                "upload_s": upload_s,
                "upload_gbps": upload_gbps,
                "step_time_s": step_time,
                "storage": storage,
                "first_checkpoint": first_checkpoint,
                "overlap_docr_push": overlap,
                "exclude_from_final": exclude,
                "exclude_reason": exclude_reason,
                "run_id": meta.get("run_id", Path(path).parent.name),
                "timestamp": s.get("wall_utc", ""),
                "precision_label": meta.get("precision_label", ""),
            })
    return rows


CKPT_FIELDS = [
    "gpu", "model", "step", "param_count", "ckpt_bytes", "bytes_per_param", "master_dtype",
    "batch_size", "tokens_per_step", "local_save_s", "upload_s", "upload_gbps", "step_time_s",
    "storage", "first_checkpoint", "overlap_docr_push", "exclude_from_final", "exclude_reason",
    "run_id", "timestamp",
]

def build_overhead(checkpoint_csv: Path, resume_csv: Path, out: Path) -> None:
    """Overhead grid: per GPU × model=8B × resume_variant(cold|lean) × sync|async.

    C/U come only from that GPU's 8B checkpoint rows (with upload_s).
    R comes only from that GPU's 8B resume rows for the matching cold/lean arm.
    1.7B has no R measurement, so it is not included in the grid.
    """
    by_gpu_model_cu: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    with checkpoint_csv.open() as f:
        for r in csv.DictReader(f):
            if str(r.get("exclude_from_final", "")).lower() in ("true", "1", "yes"):
                continue
            if r.get("local_save_s") in ("", "NOT MEASURED", None):
                continue
            if r.get("upload_s") in ("", "NOT MEASURED", None):
                continue
            model = r.get("model", "")
            if model != "8B":
                continue
            by_gpu_model_cu[(r["gpu"], model)].append(r)

    by_gpu_rv_r: Dict[tuple, List[float]] = defaultdict(list)
    if resume_csv.exists():
        with resume_csv.open() as f:
            for r in csv.DictReader(f):
                if r.get("R_s") in ("", "NOT MEASURED"):
                    continue
                if str(r.get("exclude_from_final", "")).lower() in ("true", "1"):
                    continue
                if r.get("loss_match") in ("False", "false", "0"):
                    continue
                if r.get("model") != "8B":
                    continue
                raw = (r.get("variant") or "cold").lower()
                resume_variant = "lean" if raw in ("snapshot", "lean") else "cold"
                by_gpu_rv_r[(r["gpu"], resume_variant)].append(float(r["R_s"]))

    I_min_list = [10, 30, 60, 120]
    lambdas = [0.01, 0.05, 0.1, 0.25]
    rows = []
    for (gpu, model), ckrows in sorted(by_gpu_model_cu.items()):
        local = [float(r["local_save_s"]) for r in ckrows]
        ups = [float(r["upload_s"]) for r in ckrows]
        C_local = median_or_none(local) or 0.0
        U = median_or_none(ups) or 0.0
        C_local_h = C_local / 3600.0
        U_h_full = U / 3600.0
        for resume_variant in ("cold", "lean"):
            R = median_or_none(by_gpu_rv_r.get((gpu, resume_variant), [])) or 0.0
            R_h = R / 3600.0
            for sync_async, C_h, U_var in [
                ("sync", C_local_h + U_h_full, 0.0),
                ("async", C_local_h, U_h_full),
            ]:
                for I_min in I_min_list:
                    I_h = I_min / 60.0
                    for lam in lambdas:
                        overhead = C_h / I_h + lam * (I_h / 2.0 + U_var + R_h)
                        be = overhead / (1.0 + overhead) if overhead >= 0 else float("nan")
                        I_star_min = ""
                        if lam > 0 and C_h > 0:
                            I_star_min = math.sqrt(2.0 * C_h / lam) * 60.0
                        rows.append({
                            "gpu": gpu,
                            "model": model,
                            "resume_variant": resume_variant,
                            "variant": sync_async,
                            "I_min": I_min,
                            "lambda": lam,
                            "overhead_pct": overhead * 100.0,
                            "break_even_discount_pct": be * 100.0,
                            "I_star_min": I_star_min,
                            "C_s": C_h * 3600.0,
                            "U_s": U_var * 3600.0,
                            "R_s": R,
                            "C_h": C_h,
                            "U_h": U_var,
                            "R_h": R_h,
                            "C_local_s": C_local,
                            "U_full_s": U,
                            "lambda_assumed": True,
                        })
    write_csv(out, [
        "gpu", "model", "resume_variant", "variant", "I_min", "lambda",
        "overhead_pct", "break_even_discount_pct", "I_star_min",
        "C_s", "U_s", "R_s", "C_h", "U_h", "R_h",
        "C_local_s", "U_full_s", "lambda_assumed",
    ], rows)


def try_charts(results: Path, charts: Path) -> None:
    charts.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (charts / "CHARTS_NOT_GENERATED.txt").write_text(f"matplotlib unavailable: {e}\n")
        return

    # Checkpoint save vs size
    ck = results / "checkpoint_save.csv"
    if ck.exists():
        rows = list(csv.DictReader(ck.open()))
        if rows and any(r.get("ckpt_bytes") for r in rows):
            fig, ax = plt.subplots()
            for gpu in sorted({r["gpu"] for r in rows}):
                g = [r for r in rows if r["gpu"] == gpu and r.get("ckpt_bytes") and r.get("local_save_s")]
                xs = [float(r["ckpt_bytes"]) / 1e9 for r in g]
                ys = [float(r["local_save_s"]) for r in g]
                ax.scatter(xs, ys, label=gpu)
            ax.set_xlabel("Checkpoint size (GB)")
            ax.set_ylabel("Local save time (s)")
            ax.set_title("Checkpoint save time vs checkpoint size")
            ax.legend()
            fig.savefig(charts / "checkpoint_save_vs_size.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    # Resume breakdown
    rs = results / "resume.csv"
    if rs.exists():
        rows = list(csv.DictReader(rs.open()))
        rows = [r for r in rows if r.get("R_s") not in ("", "NOT MEASURED")]
        if rows:
            fig, ax = plt.subplots()
            phases = ["provision_s", "env_s", "download_s", "pip_install_s", "load_s", "first_step_s"]
            gpus = sorted({r["gpu"] for r in rows})
            import numpy as np
            x = np.arange(len(gpus))
            bottom = np.zeros(len(gpus))
            for ph in phases:
                vals = []
                for g in gpus:
                    gs = [float(r[ph]) for r in rows if r["gpu"] == g and r.get(ph) not in ("", None)]
                    vals.append(statistics.median(gs) if gs else 0.0)
                ax.bar(x, vals, bottom=bottom, label=ph)
                bottom += np.array(vals)
            ax.set_xticks(x)
            ax.set_xticklabels(gpus)
            ax.set_ylabel("Seconds")
            ax.set_title("Simulated resume time breakdown by phase")
            ax.legend()
            fig.savefig(charts / "resume_breakdown.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    # Break-even vs I (8B only; one chart per gpu × resume_variant × sync|async)
    oh = results / "overhead_model.csv"
    if oh.exists():
        rows = list(csv.DictReader(oh.open()))
        for gpu in sorted({r["gpu"] for r in rows}):
            for resume_variant in sorted({r.get("resume_variant", "cold") for r in rows if r["gpu"] == gpu}):
                for variant in ("sync", "async"):
                    fig, ax = plt.subplots()
                    sub = [
                        r for r in rows
                        if r["gpu"] == gpu
                        and r.get("resume_variant", "cold") == resume_variant
                        and r["variant"] == variant
                    ]
                    if not sub:
                        plt.close(fig)
                        continue
                    for lam in sorted({r["lambda"] for r in sub}):
                        pts = sorted(
                            [r for r in sub if r["lambda"] == lam],
                            key=lambda r: float(r["I_min"]),
                        )
                        ax.plot(
                            [float(p["I_min"]) for p in pts],
                            [float(p["break_even_discount_pct"]) for p in pts],
                            label=f"lambda={lam}",
                        )
                    ax.set_xlabel("Checkpoint interval I (minutes)")
                    ax.set_ylabel("Break-even spot discount (%)")
                    ax.set_title(
                        f"{gpu} 8B {resume_variant} {variant}: break-even vs I"
                    )
                    ax.legend()
                    fig.savefig(
                        charts / f"breakeven_{gpu}_{resume_variant}_{variant}.png",
                        dpi=140,
                        bbox_inches="tight",
                    )
                    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="evidence root")
    args = ap.parse_args()
    root = Path(args.root)
    results = root / "results"
    results.mkdir(exist_ok=True)

    # Optional run metadata files: logs/**/run_meta.json
    meta_by_events: Dict[str, Dict[str, Any]] = {}
    for meta_path in (root / "logs").rglob("run_meta.json"):
        meta = json.loads(meta_path.read_text())
        ev = meta_path.parent / "events.jsonl"
        meta_by_events[str(ev)] = meta

    events = load_events(root / "logs")
    ck_rows = build_checkpoint_rows(events, meta_by_events)
    # Exclude smoke/bf16 from final artifact note; still write them with flag
    if not ck_rows:
        ck_rows = [{
            "gpu": "ALL", "model": "", "step": "", "ckpt_bytes": "NOT MEASURED",
            "param_count": "NOT MEASURED", "bytes_per_param": "NOT MEASURED",
            "master_dtype": "", "batch_size": "", "tokens_per_step": "",
            "local_save_s": "NOT MEASURED", "upload_s": "NOT MEASURED",
            "upload_gbps": "NOT MEASURED", "step_time_s": "NOT MEASURED",
            "storage": "", "first_checkpoint": "", "overlap_docr_push": "",
            "exclude_from_final": True,
            "exclude_reason": "no events", "run_id": "", "timestamp": "",
            "precision_label": "",
        }]
    write_csv(results / "checkpoint_save.csv", CKPT_FIELDS, ck_rows)

    # Overhead model uses only non-excluded rows
    ck_final = results / "checkpoint_save_final.csv"
    final_rows = [r for r in ck_rows if str(r.get("exclude_from_final")).lower() not in ("true", "1")]
    write_csv(ck_final, CKPT_FIELDS,
              final_rows if final_rows else [{
                  "gpu": "", "model": "", "step": "", "ckpt_bytes": "NOT MEASURED",
                  "param_count": "", "bytes_per_param": "", "master_dtype": "fp32",
                  "batch_size": "", "tokens_per_step": "",
                  "local_save_s": "NOT MEASURED", "upload_s": "NOT MEASURED",
                  "upload_gbps": "", "step_time_s": "", "storage": "",
                  "first_checkpoint": "", "overlap_docr_push": "",
                  "exclude_from_final": False, "exclude_reason": "awaiting full runs",
                  "run_id": "", "timestamp": "",
              }])


    # Resume rows from resume_result.json files (keep excluded boot_disk rows for comparison)
    resume_rows = []
    for p in (root / "logs").rglob("resume_result.json"):
        r = json.loads(p.read_text())
        meta = {}
        mp = p.parent / "run_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        if r.get("first_step_s") in (None, ""):
            continue
        storage = meta.get("storage", "")
        if not storage and str(p.parent.name).startswith(("resume-cold-", "resume-snap-")):
            storage = "boot_disk"
        exclude = meta.get("exclude_from_final_results", False)
        exclude_reason = meta.get("exclude_reason", "")
        if storage == "boot_disk":
            exclude = True
            exclude_reason = exclude_reason or "storage_boot_disk"
        # Prefer explicit steady-state step_time; never fall back to first_step_s (that was the bug)
        step_time = r.get("step_time_s", meta.get("steady_step_time_s", meta.get("step_time_s", "")))
        try:
            if step_time not in ("", None) and r.get("first_step_s") not in ("", None):
                if abs(float(step_time) - float(r["first_step_s"])) < 1e-9:
                    step_time = meta.get("steady_step_time_s", "")
        except (TypeError, ValueError):
            pass
        image_size = r.get("image_size_bytes", meta.get("image_size_bytes", ""))
        steps_lost = r.get("steps_lost", meta.get("steps_lost", ""))
        resume_rows.append({
            "gpu": meta.get("gpu", r.get("gpu", "")),
            "model": meta.get("model", r.get("model", "")),
            "variant": r.get("variant", meta.get("variant", "cold")),
            "provision_s": r.get("provision_s", ""),
            "env_s": r.get("env_s", ""),
            "image_pull_s": r.get("image_pull_s", ""),
            "container_start_s": r.get("container_start_s", ""),
            "image_size_bytes": image_size,
            "download_s": r.get("download_s", ""),
            "pip_install_s": r.get("pip_install_s", ""),
            "load_s": r.get("load_s", ""),
            "first_step_s": r.get("first_step_s", ""),
            "R_s": r.get("R_s", ""),
            "steps_lost": steps_lost,
            "step_time_s": step_time,
            "batch_size": r.get("batch_size", meta.get("batch_size", "")),
            "capacity_retries": r.get("capacity_retries", 0),
            "retry_wait_s": r.get("retry_wait_s", 0),
            "capacity_error": r.get("capacity_error", meta.get("capacity_error", "")),
            "loss_match": r.get("loss_match", ""),
            "loss_match_resume_consistency": r.get(
                "loss_match_resume_consistency", meta.get("loss_match_resume_consistency", "")
            ),
            "loss_match_strict": r.get("loss_match_strict", meta.get("loss_match_strict", "")),
            "storage": storage,
            "run_id": p.parent.name,
            "exclude_from_final": exclude,
            "exclude_reason": exclude_reason,
            "precision_label": meta.get("precision_label", ""),
        })
    if not resume_rows:
        resume_rows = [{
            "gpu": "", "model": "", "variant": "", "provision_s": "NOT MEASURED",
            "env_s": "NOT MEASURED", "image_pull_s": "NOT MEASURED",
            "container_start_s": "NOT MEASURED", "image_size_bytes": "NOT MEASURED",
            "download_s": "NOT MEASURED", "pip_install_s": "NOT MEASURED", "load_s": "NOT MEASURED", "first_step_s": "NOT MEASURED",
            "R_s": "NOT MEASURED", "steps_lost": "", "step_time_s": "", "batch_size": "",
            "capacity_retries": "", "retry_wait_s": "", "capacity_error": "",
            "loss_match": "", "loss_match_resume_consistency": "", "loss_match_strict": "",
            "storage": "", "run_id": "",
            "exclude_from_final": True, "exclude_reason": "no resume results", "precision_label": "",
        }]
    write_csv(results / "resume.csv",
              ["gpu", "model", "variant", "provision_s", "env_s", "image_pull_s",
               "container_start_s", "image_size_bytes", "download_s", "pip_install_s", "load_s", "first_step_s", "R_s",
               "steps_lost", "step_time_s", "batch_size", "capacity_retries", "retry_wait_s", "capacity_error",
               "loss_match", "loss_match_resume_consistency", "loss_match_strict", "storage", "run_id",
               "exclude_from_final", "exclude_reason", "precision_label"],
              resume_rows)

    # Throughput placeholder
    tp = results / "throughput.csv"
    if not tp.exists():
        write_csv(tp, ["gpu", "size", "mode", "steps_per_s_mean", "steps_per_s_std",
                       "tokens_per_s_mean", "tokens_per_s_std", "reason"],
                  [{"gpu": "MI355X/B300", "size": "", "mode": "parity",
                    "steps_per_s_mean": "NOT MEASURED", "steps_per_s_std": "NOT MEASURED",
                    "tokens_per_s_mean": "NOT MEASURED", "tokens_per_s_std": "NOT MEASURED",
                    "reason": "No on-demand MI355X or B300 sizes"}])

    build_overhead(ck_final if ck_final.exists() else results / "checkpoint_save.csv",
                   results / "resume.csv", results / "overhead_model.csv")
    try_charts(results, root / "reports" / "charts")
    print("Wrote results under", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
