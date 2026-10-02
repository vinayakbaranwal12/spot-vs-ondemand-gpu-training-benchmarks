#!/usr/bin/env python3
"""
Spot vs on-demand GPU training benchmark: train, checkpoint, and resume.

Uses Qwen3 configs with randomly initialized weights, FSDP2 fully_shard,
torch.distributed.checkpoint (DCP), synthetic deterministic batches.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mono_marks: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._mono_marks[name] = time.perf_counter()

    def emit(self, event: str, **fields: Any) -> None:
        mono = time.perf_counter()
        payload = {
            "event": event,
            "wall_utc": utc_now(),
            "mono_s": mono,
            **fields,
        }
        if "since" in fields:
            # caller may pass duration_s directly
            pass
        line = json.dumps(payload, sort_keys=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def emit_span(self, event: str, start_mark: str, **fields: Any) -> float:
        start = self._mono_marks[start_mark]
        dur = time.perf_counter() - start
        self.emit(event, duration_s=dur, **fields)
        return dur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["1.7B", "8B", "32B", "Qwen3-1.7B", "Qwen3-8B", "Qwen3-32B"])
    p.add_argument("--events", required=True, help="path to events.jsonl")
    p.add_argument("--ckpt-dir", required=True, help="local scratch checkpoint root")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=0, help="micro-batch size; 0 = auto-tune to ~80-90% VRAM")
    p.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps per optimizer step")
    p.add_argument("--target-mem-frac", type=float, default=0.85, help="target GPU memory fraction for auto batch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="torch.use_deterministic_algorithms(True); requires CUBLAS_WORKSPACE_CONFIG=:4096:8 in env",
    )
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--steady-steps", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=20, help="steps between checkpoints after steady")
    p.add_argument("--num-ckpts", type=int, default=3)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--resume-from", default="", help="local path or spaces URI already downloaded")
    p.add_argument("--kill-at-step", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="hard stop (0=auto from ckpt plan)")
    p.add_argument("--smoke", action="store_true", help="1.7B smoke: fewer steps, 1 ckpt")
    p.add_argument(
        "--master-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Parameter and AdamW state dtype. Full runs use fp32 (~12-16 B/param). Smoke used bf16.",
    )
    p.add_argument("--upload-after-ckpt", action="store_true")
    p.add_argument("--spaces-uri", default="", help="s3://bucket/prefix for uploads")
    p.add_argument("--spaces-endpoint", default="")
    p.add_argument("--hf-id", default="", help="override HF config id")
    return p.parse_args()


def model_to_hf(model: str) -> str:
    m = model.replace("Qwen3-", "")
    return {
        "1.7B": "Qwen/Qwen3-1.7B",
        "8B": "Qwen/Qwen3-8B",
        "32B": "Qwen/Qwen3-32B",
    }[m]


def ensure_transformers_qwen3() -> str:
    import transformers

    ver = transformers.__version__
    # Qwen3 support landed in transformers 4.51+ (approximate); upgrade if import config fails later
    return ver


def make_batch(seed: int, step: int, batch_size: int, seq_len: int, vocab_size: int, device):
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(seed + step * 10007)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.long)
    labels = input_ids.clone()
    return input_ids.to(device), labels.to(device)


def auto_tune_batch_size(
    model,
    optim,
    *,
    seed: int,
    seq_len: int,
    vocab: int,
    device,
    world_size: int,
    target_frac: float,
    grad_accum: int,
    events: EventLog,
    rank: int,
) -> int:
    """Find largest micro-batch that keeps peak VRAM near target_frac after a short probe."""
    import torch

    total = torch.cuda.get_device_properties(device).total_memory
    # Conservative start; double until OOM or over target, then back off.
    candidates = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    probes: List[Dict[str, Any]] = []
    best = 1
    model.train()
    if rank == 0:
        events.emit(
            "batch_tune_start",
            candidates=candidates,
            target_mem_frac=target_frac,
            total_vram_bytes=total,
            grad_accum=grad_accum,
            seq_len=seq_len,
        )
    for bs in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            optim.zero_grad(set_to_none=True)
            for micro in range(max(1, min(grad_accum, 2))):
                input_ids, labels = make_batch(seed, micro, bs, seq_len, vocab, device)
                out = model(input_ids=input_ids, labels=labels)
                (out.loss / grad_accum).backward()
            optim.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            frac = peak / total if total else 0.0
            probe = {
                "batch_size": bs,
                "peak_bytes": int(peak),
                "peak_gib": round(peak / (1024**3), 3),
                "mem_frac": frac,
                "total_bytes": int(total),
                "status": "ok" if frac <= target_frac else "over_target",
            }
            probes.append(probe)
            if rank == 0:
                events.emit("batch_probe", **probe)
            if frac <= target_frac:
                best = bs
            else:
                break
        except torch.cuda.OutOfMemoryError:
            probe = {"batch_size": bs, "status": "oom", "total_bytes": int(total)}
            probes.append(probe)
            if rank == 0:
                events.emit("batch_probe_oom", **probe)
            torch.cuda.empty_cache()
            break
    if rank == 0:
        events.emit(
            "batch_selected",
            batch_size=best,
            grad_accum=grad_accum,
            tokens_per_step=best * grad_accum * seq_len * world_size,
            target_mem_frac=target_frac,
            auto=True,
            source="auto_tune",
            probes=probes,
            probes_tried=[p["batch_size"] for p in probes],
        )
    return best


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.warmup_steps = min(args.warmup_steps, 5)
        args.steady_steps = min(args.steady_steps, 20)
        args.num_ckpts = 1
        args.ckpt_every = 15
        args.log_every = 1

    events = EventLog(Path(args.events))
    events.mark("init")
    events.emit("init_start", smoke=bool(args.smoke), model=args.model)

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
        StateDictOptions,
    )

    # torchrun sets these
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    backend = "nccl"
    if torch.version.hip is not None:
        # ROCm uses nccl-compatible RCCL via nccl backend name in recent torch
        backend = "nccl"

    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Weight init + CUDA RNG must be seeded for from-scratch run-to-run compares.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        # CUBLAS_WORKSPACE_CONFIG must already be set in the process environment.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if rank == 0:
            events.emit(
                "deterministic_mode",
                enabled=True,
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                seed=args.seed,
            )

    hf_id = args.hf_id or model_to_hf(args.model)
    tf_ver = ensure_transformers_qwen3()

    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        if rank == 0:
            print(f"config load failed ({e}); upgrading transformers", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "transformers"])
        from transformers import AutoConfig, AutoModelForCausalLM  # noqa: F811
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        import transformers as _tf
        tf_ver = _tf.__version__

    # Reduce seq if needed later; record actual
    seq_len = args.seq_len
    # Full runs: fp32 master weights + AdamW state; bf16 compute via FSDP2 MixedPrecisionPolicy.
    # Smoke historically used bf16 params (~6 B/param); do not use those for final results.
    master_dtype = torch.float32 if args.master_dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_config(config, torch_dtype=master_dtype)
    model.to(device=device, dtype=master_dtype)
    model.train()

    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # shard transformer blocks if present
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Parameter count for bytes/param reporting
    param_count = sum(p.numel() for p in model.parameters())

    start_step = 0
    ckpt_root = Path(args.ckpt_dir)
    if rank == 0:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    dist.barrier()
    events.emit_span(
        "model_ready",
        "init",
        rank=rank,
        world_size=world_size,
        transformers=tf_ver,
        torch=torch.__version__,
        hf_id=hf_id,
        seq_len=seq_len,
        smoke=bool(args.smoke),
        master_dtype=args.master_dtype,
        param_count=int(param_count),
        precision_note=(
            "smoke, bf16 state" if args.master_dtype == "bf16" else "fp32 master + AdamW; bf16 compute"
        ),
    )

    if args.resume_from:
        resume_path = Path(args.resume_from)
        events.mark("ckpt_load")
        events.emit("ckpt_load_start", path=str(resume_path))
        app_state = {
            "model": model,
            "optim": optim,
        }
        # Load step meta
        meta_path = resume_path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            start_step = int(meta["step"]) + 1
            if "torch_rng_hex" in meta:
                torch.set_rng_state(
                    torch.tensor(bytearray.fromhex(meta["torch_rng_hex"]), dtype=torch.uint8)
                )
            if "cuda_rng_hex" in meta and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [torch.tensor(bytearray.fromhex(h), dtype=torch.uint8) for h in meta["cuda_rng_hex"]]
                )
            if "python_random_state" in meta:
                ver = int(meta.get("python_random_version", 3))
                random.setstate((ver, tuple(meta["python_random_state"]), None))
            if "numpy_rng_bit_generator_state" in meta:
                try:
                    import numpy as np

                    st = (
                        "MT19937",
                        np.array(meta["numpy_rng_bit_generator_state"], dtype=np.uint32),
                        int(meta.get("numpy_rng_pos", 624)),
                        0,
                        0.0,
                    )
                    np.random.set_state(st)
                except Exception as e:
                    if rank == 0:
                        print(f"numpy rng restore skipped: {e}", flush=True)

        options = StateDictOptions(full_state_dict=False)
        model_sd = get_model_state_dict(model, options=options)
        optim_sd = get_optimizer_state_dict(model, optim, options=options)
        state_dict = {"model": model_sd, "optim": optim_sd}
        dcp.load(state_dict=state_dict, checkpoint_id=str(resume_path / "dcp"))
        set_model_state_dict(model, model_sd, options=options)
        set_optimizer_state_dict(model, optim, optim_sd, options=options)
        load_s = events.emit_span("ckpt_load_end", "ckpt_load", path=str(resume_path), resume_step=start_step)
        events.emit("first_step_after_resume", pending_step=start_step)

    vocab = getattr(config, "vocab_size", 151936)
    grad_accum = max(1, int(args.grad_accum))

    # Resume must reuse original batch_size / grad_accum. Never auto-tune on resume.
    resume_meta: Dict[str, Any] = {}
    if args.resume_from:
        meta_path = Path(args.resume_from) / "meta.json"
        if meta_path.exists():
            resume_meta = json.loads(meta_path.read_text())

    if args.resume_from:
        if args.batch_size and args.batch_size > 0:
            batch_size = int(args.batch_size)
        elif resume_meta.get("batch_size"):
            batch_size = int(resume_meta["batch_size"])
        else:
            raise SystemExit(
                "RESUME_BATCH_REQUIRED: pass --batch-size from original run_meta.json "
                "(or ensure checkpoint meta.json has batch_size). Auto-tune is disabled on resume."
            )
        if resume_meta.get("grad_accum"):
            grad_accum = int(resume_meta["grad_accum"])
        elif args.grad_accum:
            grad_accum = max(1, int(args.grad_accum))
        if rank == 0:
            events.emit(
                "batch_selected",
                batch_size=batch_size,
                grad_accum=grad_accum,
                tokens_per_step=batch_size * grad_accum * seq_len * world_size,
                auto=False,
                source="resume_meta_or_cli",
                resume_meta_batch_size=resume_meta.get("batch_size"),
                resume_meta_grad_accum=resume_meta.get("grad_accum"),
            )
    elif args.batch_size and args.batch_size > 0:
        batch_size = int(args.batch_size)
        if rank == 0:
            events.emit(
                "batch_selected",
                batch_size=batch_size,
                grad_accum=grad_accum,
                tokens_per_step=batch_size * grad_accum * seq_len * world_size,
                target_mem_frac=args.target_mem_frac,
                auto=False,
                source="cli",
            )
    else:
        batch_size = auto_tune_batch_size(
            model,
            optim,
            seed=args.seed,
            seq_len=seq_len,
            vocab=vocab,
            device=device,
            world_size=world_size,
            target_frac=args.target_mem_frac,
            grad_accum=grad_accum,
            events=events,
            rank=rank,
        )
        # Broadcast selected batch size
        if world_size > 1:
            t = torch.tensor([batch_size], device=device, dtype=torch.int64)
            dist.broadcast(t, src=0)
            batch_size = int(t.item())

    tokens_per_step = batch_size * grad_accum * seq_len * world_size
    if rank == 0:
        events.emit(
            "train_hparams",
            batch_size=batch_size,
            grad_accum=grad_accum,
            seq_len=seq_len,
            tokens_per_step=tokens_per_step,
            master_dtype=args.master_dtype,
        )

    # Plan steps. On resume, --max-steps means additional steps from start_step
    # (so resume probes can use --max-steps 3 after a large checkpoint step).
    if args.max_steps > 0:
        if args.resume_from:
            max_steps = start_step + args.max_steps
        else:
            max_steps = args.max_steps
    else:
        # warmup + steady + num_ckpts * ckpt_every + a few after last
        max_steps = args.warmup_steps + args.steady_steps + args.num_ckpts * args.ckpt_every + 5

    ckpts_done = 0
    first_ckpt_after = args.warmup_steps + args.steady_steps
    last_losses: Dict[int, float] = {}

    step = start_step
    while step < max_steps:
        if args.kill_at_step and step >= args.kill_at_step:
            events.emit("killed", step=step, reason="kill_at_step")
            break

        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        loss_v = 0.0
        for micro in range(grad_accum):
            input_ids, labels = make_batch(
                args.seed, step * grad_accum + micro, batch_size, seq_len, vocab, device
            )
            out = model(input_ids=input_ids, labels=labels)
            (out.loss / grad_accum).backward()
            loss_v = float(out.loss.detach().float().cpu())
        optim.step()
        torch.cuda.synchronize()
        step_t = time.perf_counter() - t0
        last_losses[step] = loss_v
        tokens = tokens_per_step
        tok_s = tokens / step_t if step_t > 0 else 0.0
        mem_frac = (
            torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
        )

        if step % args.log_every == 0 or step == start_step:
            events.emit(
                "step",
                step=step,
                loss=loss_v,
                step_time_s=step_t,
                tokens_per_s=tok_s,
                tokens=tokens,
                tokens_per_step=tokens_per_step,
                batch_size=batch_size,
                grad_accum=grad_accum,
                mem_frac=mem_frac,
            )

        # Checkpoint schedule: after first_ckpt_after, every ckpt_every, until num_ckpts
        should_ckpt = (
            step >= first_ckpt_after
            and ckpts_done < args.num_ckpts
            and ((step - first_ckpt_after) % args.ckpt_every == 0)
        )
        if should_ckpt:
            ckpt_name = f"ckpt-step-{step}"
            local_dir = ckpt_root / ckpt_name
            if rank == 0:
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)
                # free space
                du = shutil.disk_usage(str(ckpt_root))
                events.emit("ckpt_disk", free_bytes=du.free, path=str(ckpt_root))

            dist.barrier()
            events.mark("ckpt_save")
            events.emit("ckpt_save_start", step=step, path=str(local_dir))

            options = StateDictOptions(full_state_dict=False)
            model_sd = get_model_state_dict(model, options=options)
            optim_sd = get_optimizer_state_dict(model, optim, options=options)
            state_dict = {"model": model_sd, "optim": optim_sd}
            dcp.save(state_dict=state_dict, checkpoint_id=str(local_dir / "dcp"))

            if rank == 0:
                meta = {
                    "step": step,
                    "loss": loss_v,
                    "model": args.model,
                    "hf_id": hf_id,
                    "seq_len": seq_len,
                    "seed": args.seed,
                    "master_dtype": args.master_dtype,
                    "param_count": int(param_count),
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "tokens_per_step": tokens_per_step,
                    "torch_rng_hex": torch.get_rng_state().numpy().tobytes().hex(),
                    "cuda_rng_hex": [s.numpy().tobytes().hex() for s in torch.cuda.get_rng_state_all()],
                    "python_random_state": list(random.getstate()[1]),
                    "python_random_version": int(random.getstate()[0]),
                }
                try:
                    import numpy as np

                    meta["numpy_rng_bit_generator_state"] = np.random.get_state()[1].tolist()
                    meta["numpy_rng_pos"] = int(np.random.get_state()[2])
                except Exception:
                    pass
                # Optimizer step counter when present (AdamW)
                try:
                    opt_state = optim.state_dict()
                    meta["optimizer_state_steps"] = {
                        str(k): int(v["step"])
                        for k, v in opt_state.get("state", {}).items()
                        if isinstance(v, dict) and "step" in v
                    }
                    # keep a compact sample only
                    if len(meta["optimizer_state_steps"]) > 8:
                        ks = sorted(meta["optimizer_state_steps"], key=int)[:4] + sorted(
                            meta["optimizer_state_steps"], key=int
                        )[-4:]
                        meta["optimizer_state_steps"] = {k: meta["optimizer_state_steps"][k] for k in ks}
                except Exception:
                    pass
                (local_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                # size
                raw = subprocess.check_output(["du", "-sb", str(local_dir)], text=True)
                ckpt_bytes = int(raw.split()[0])
            else:
                ckpt_bytes = 0

            dist.barrier()
            if world_size > 1:
                t = torch.tensor([ckpt_bytes], device=device)
                dist.broadcast(t, src=0)
                ckpt_bytes = int(t.item())

            bytes_per_param = (float(ckpt_bytes) / float(param_count)) if param_count else None
            local_save_s = events.emit_span(
                "ckpt_save_end",
                "ckpt_save",
                step=step,
                path=str(local_dir),
                ckpt_bytes=ckpt_bytes,
                param_count=int(param_count),
                bytes_per_param=bytes_per_param,
                master_dtype=args.master_dtype,
                batch_size=batch_size,
                grad_accum=grad_accum,
                tokens_per_step=tokens_per_step,
            )

            if args.upload_after_ckpt and args.spaces_uri and rank == 0:
                events.mark("upload")
                events.emit("upload_start", path=str(local_dir), spaces_uri=args.spaces_uri)
                endpoint = args.spaces_endpoint
                dest = args.spaces_uri.rstrip("/") + "/" + ckpt_name + "/"
                s5cmd_bin = "/host-bin/s5cmd" if Path("/host-bin/s5cmd").is_file() else (shutil.which("s5cmd") or "")
                if not s5cmd_bin:
                    # Fall back: write marker for host uploader; do not fail the train process mid-run
                    marker = local_dir.parent / f".need_upload_{ckpt_name}"
                    marker.write_text(json.dumps({"path": str(local_dir), "dest": dest, "endpoint": endpoint}))
                    events.emit("upload_deferred_to_host", path=str(local_dir), dest=dest)
                    ckpts_done += 1
                    step += 1
                    continue
                if not Path(s5cmd_bin).exists() and s5cmd_bin == "/host-bin/s5cmd":
                    raise FileNotFoundError(
                        "s5cmd not found in container; mount host binary at /host-bin/s5cmd"
                    )
                cmd = [
                    s5cmd_bin,
                    "--endpoint-url", endpoint,
                    "cp",
                    str(local_dir) + "/",
                    dest,
                ]
                env = os.environ.copy()
                subprocess.check_call(cmd, env=env)
                upload_s = events.emit_span("upload_end", "upload", ckpt_bytes=ckpt_bytes, dest=dest)
                # verify listing non-empty
                subprocess.check_call(
                    [s5cmd_bin, "--endpoint-url", endpoint, "ls", dest],
                    env=env,
                )
                # delete local after verified upload
                shutil.rmtree(local_dir)
                events.emit(
                    "ckpt_uploaded_and_local_deleted",
                    step=step,
                    local_save_s=local_save_s,
                    upload_s=upload_s,
                    ckpt_bytes=ckpt_bytes,
                    dest=dest,
                )
            ckpts_done += 1

        step += 1

    # Write loss table for resume verification
    if rank == 0:
        loss_path = Path(args.events).with_name("losses.json")
        loss_path.write_text(json.dumps({str(k): v for k, v in last_losses.items()}, indent=2))

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
