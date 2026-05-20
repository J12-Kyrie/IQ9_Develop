#!/home/ubuntu/miniforge3/bin/python
"""Run repeatable Qwen3-VL Genie benchmarks on the target device.

The runner is intentionally target-side and self-contained. It executes the
same Genie command as the production Qwen3-VL script, writes every run artifact
under a timestamped directory, samples host/DSP daemon memory with psutil, and
summarizes Genie profile JSON metrics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import pty
import select
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import psutil
except ImportError as exc:  # pragma: no cover - target dependency guard
    raise SystemExit(
        "psutil is required. Run this with /home/ubuntu/miniforge3/bin/python "
        "on the target device."
    ) from exc


DEPLOY_ROOT = Path("/mnt/workspace/develop/qwen3vl")
DEFAULT_OUTPUT_ROOT = DEPLOY_ROOT / "profiles"
DEFAULT_INDICES = "0,7,14,19"
DEFAULT_EMBED_SCALE = "0.00037788687041029334"
DEFAULT_EMBED_OFFSET = "-36275"
DSP_DAEMON_NAMES = ("cdsprpcd", "adsprpcd")
SUMMARY_FIELDS = [
    "idx",
    "rep",
    "status",
    "returncode",
    "profile_json",
    "stdout",
    "samples_csv",
    "init_time_ms",
    "query_duration_ms",
    "prompt_tokens",
    "prompt_processing_rate",
    "ttft_ms",
    "generated_tokens",
    "token_generation_time_ms",
    "token_generation_rate",
    "ssd_acceptance_rate",
    "peak_rss_bytes",
    "peak_pss_bytes",
    "regression_candidate",
    "error",
]


def parse_indices(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("at least one index is required")
    return values


def local_timestamp() -> Tuple[str, str, float, str]:
    now = dt.datetime.now().astimezone()
    return (
        now.strftime("%Y%m%d_%H%M%S"),
        now.isoformat(),
        now.timestamp(),
        now.tzname() or "",
    )


def mkdirs(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "runs": root / "runs",
        "samples": root / "samples",
        "snapshot": root / "config_snapshot",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_capture(cmd: Sequence[str], cwd: Path) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {
            "cmd": list(cmd),
            "returncode": result.returncode,
            "output": result.stdout.strip(),
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {"cmd": list(cmd), "returncode": None, "error": str(exc)}


def collect_metadata(deploy_root: Path) -> Dict[str, Any]:
    stamp, iso, epoch, timezone = local_timestamp()
    metadata: Dict[str, Any] = {
        "created_timestamp": stamp,
        "device_time_iso": iso,
        "device_epoch_time": epoch,
        "timezone": timezone,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uname": " ".join(platform.uname()),
        "psutil_version": getattr(psutil, "__version__", None),
        "deploy_root": str(deploy_root),
        "commands": {},
    }
    command_map = {
        "genie_t2t_run_help": [str(deploy_root / "bin" / "genie-t2t-run"), "--help"],
        "qnn_net_run_version": ["qnn-net-run", "--version"],
        "qnn_profile_viewer_version": ["qnn-profile-viewer", "--version"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
        "dev_fastrpc": ["sh", "-c", "ls -l /dev/fastrpc* 2>/dev/null"],
    }
    for name, cmd in command_map.items():
        metadata["commands"][name] = run_capture(cmd, deploy_root)
    return metadata


def copy_if_exists(src: Path, dst_dir: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def snapshot_configs(deploy_root: Path, snapshot_dir: Path) -> Dict[str, Any]:
    copied: Dict[str, Any] = {"files": [], "missing": []}
    candidates = [deploy_root / "qwen3vl.json", deploy_root / "run_qwen3vl.sh"]
    candidates.extend(sorted(deploy_root.glob("htp_backend_ext_config*.json")))
    candidates.extend(sorted((deploy_root / "models" / "data").glob("htp_backend_ext_config*.json")))
    seen = set()
    for src in candidates:
        if src in seen:
            continue
        seen.add(src)
        copied_path = copy_if_exists(src, snapshot_dir)
        if copied_path is None:
            copied["missing"].append(str(src))
        else:
            copied["files"].append(copied_path)
    return copied


def flatten_ctx_bins(ctx_bins: Any) -> List[str]:
    flattened: List[str] = []
    if isinstance(ctx_bins, list):
        for item in ctx_bins:
            if isinstance(item, list):
                flattened.extend(str(subitem) for subitem in item)
            else:
                flattened.append(str(item))
    return flattened


def build_case_config(
    deploy_root: Path,
    case_name: str,
    snapshot_dir: Path,
    explicit_config: Optional[Path],
) -> Tuple[Path, Dict[str, Any]]:
    source_config = explicit_config or (deploy_root / "qwen3vl.json")
    with source_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    case_info: Dict[str, Any] = {
        "source_config": str(source_config),
        "runtime_config": str(source_config),
        "case_transform": "none",
    }
    if case_name == "dual_cdsp_baseline":
        copy_if_exists(source_config, snapshot_dir)
        return source_config, case_info

    engine = config.get("dialog", {}).get("engine", {})
    backend = engine.get("backend", {})
    model_binary = engine.get("model", {}).get("binary", {})

    if case_name == "single_cdsp_device0":
        extensions = backend.get("extensions")
        if isinstance(extensions, list):
            backend["extensions"] = extensions[:1]
        ctx_bins = model_binary.get("ctx-bins")
        flattened = flatten_ctx_bins(ctx_bins)
        if flattened:
            model_binary["ctx-bins"] = [flattened]
        case_info["case_transform"] = "single extension, flattened ctx-bins"
    elif case_name == "swapped_cdsp_order":
        extensions = backend.get("extensions")
        if isinstance(extensions, list) and len(extensions) >= 2:
            backend["extensions"] = list(reversed(extensions))
        else:
            case_info["warning"] = "extensions is not a list with at least two entries"
        case_info["case_transform"] = "reversed extensions order"
    else:
        copy_if_exists(source_config, snapshot_dir)
        return source_config, case_info

    runtime_config = snapshot_dir / f"qwen3vl_profile_{case_name}.json"
    with runtime_config.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    case_info["runtime_config"] = str(runtime_config)
    return runtime_config, case_info


def process_identity(proc: psutil.Process) -> Tuple[str, str]:
    try:
        name = proc.name()
    except psutil.Error:
        name = ""
    try:
        cmdline = " ".join(proc.cmdline())
    except psutil.Error:
        cmdline = ""
    return name, cmdline


def find_daemon_processes() -> List[psutil.Process]:
    daemons: List[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(daemon in name or daemon in cmdline for daemon in DSP_DAEMON_NAMES):
            daemons.append(proc)
    return daemons


def safe_mem(proc: psutil.Process) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        full = proc.memory_full_info()
        rss = getattr(full, "rss", None)
        pss = getattr(full, "pss", None)
        uss = getattr(full, "uss", None)
        return rss, pss if pss is not None else rss, uss
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        try:
            rss = proc.memory_info().rss
            return rss, rss, None
        except psutil.Error:
            return None, None, None


def safe_io(proc: psutil.Process) -> Tuple[Optional[int], Optional[int]]:
    try:
        counters = proc.io_counters()
        return getattr(counters, "read_bytes", None), getattr(counters, "write_bytes", None)
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return None, None


def collect_processes(root_pid: int) -> List[Tuple[str, psutil.Process]]:
    collected: List[Tuple[str, psutil.Process]] = []
    seen = set()
    try:
        root = psutil.Process(root_pid)
        root_and_children = [root] + root.children(recursive=True)
    except psutil.Error:
        root_and_children = []
    for proc in root_and_children:
        if proc.pid not in seen:
            collected.append(("run_tree", proc))
            seen.add(proc.pid)
    for proc in find_daemon_processes():
        if proc.pid not in seen:
            collected.append(("dsp_daemon", proc))
            seen.add(proc.pid)
    return collected


class ProcessSampler:
    def __init__(self, root_pid: int, sample_sec: float, output_csv: Path):
        self.root_pid = root_pid
        self.sample_sec = sample_sec
        self.output_csv = output_csv
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss = 0
        self.peak_pss = 0
        self.seen_names: List[str] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.sample_sec * 4))

    def _run(self) -> None:
        fieldnames = [
            "timestamp_iso",
            "epoch",
            "kind",
            "pid",
            "name",
            "cmdline",
            "cpu_percent",
            "rss_bytes",
            "pss_bytes",
            "uss_bytes",
            "num_threads",
            "read_bytes",
            "write_bytes",
        ]
        cpu_warmed = set()
        with self.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            while not self.stop_event.is_set():
                now = dt.datetime.now().astimezone()
                for kind, proc in collect_processes(self.root_pid):
                    try:
                        if proc.pid not in cpu_warmed:
                            proc.cpu_percent(interval=None)
                            cpu_warmed.add(proc.pid)
                        name, cmdline = process_identity(proc)
                        rss, pss, uss = safe_mem(proc)
                        read_bytes, write_bytes = safe_io(proc)
                        try:
                            threads = proc.num_threads()
                        except psutil.Error:
                            threads = None
                        try:
                            cpu = proc.cpu_percent(interval=None)
                        except psutil.Error:
                            cpu = None
                        if rss is not None:
                            self.peak_rss = max(self.peak_rss, rss)
                        if pss is not None:
                            self.peak_pss = max(self.peak_pss, pss)
                        if name and name not in self.seen_names:
                            self.seen_names.append(name)
                        writer.writerow(
                            {
                                "timestamp_iso": now.isoformat(),
                                "epoch": now.timestamp(),
                                "kind": kind,
                                "pid": proc.pid,
                                "name": name,
                                "cmdline": cmdline,
                                "cpu_percent": cpu,
                                "rss_bytes": rss,
                                "pss_bytes": pss,
                                "uss_bytes": uss,
                                "num_threads": threads,
                                "read_bytes": read_bytes,
                                "write_bytes": write_bytes,
                            }
                        )
                    except psutil.NoSuchProcess:
                        continue
                handle.flush()
                self.stop_event.wait(self.sample_sec)


def start_sidecar(command: Sequence[str], output_path: Path, cwd: Path) -> Optional[subprocess.Popen]:
    if shutil.which(command[0]) is None:
        output_path.write_text(f"{command[0]} not found\n", encoding="utf-8")
        return None
    handle = output_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        handle.write(f"failed to start {' '.join(command)}: {exc}\n")
        handle.close()
        return None


def stop_sidecar(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:  # pylint: disable=broad-except
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # pylint: disable=broad-except
            pass


def collect_journal(output_path: Path, since_iso: str, cwd: Path) -> None:
    if shutil.which("journalctl") is None:
        output_path.write_text("journalctl not found\n", encoding="utf-8")
        return
    cmd = ["journalctl", "-b", "--since", since_iso]
    result = run_capture(cmd, cwd)
    text = result.get("output") or ""
    lines = [
        line
        for line in text.splitlines()
        if any(term in line.lower() for term in ("genie", "qnn", "htp", "fastrpc", "cdsp"))
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def metric_value(event: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = event.get(name)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        if isinstance(value, (int, float)):
            return value
    return None


def convert_us_to_ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value / 1000.0


def parse_genie_profile(profile_path: Path) -> Dict[str, Any]:
    with profile_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    events: List[Dict[str, Any]] = []
    for component in data.get("components", []):
        if isinstance(component, dict):
            events.extend(event for event in component.get("events", []) if isinstance(event, dict))
    create_event = next((event for event in events if event.get("type") == "GenieDialog_create"), {})
    query_event = next((event for event in events if event.get("type") == "GenieDialog_query"), {})
    return {
        "init_time_ms": convert_us_to_ms(metric_value(create_event, "init-time", "init_time")),
        "query_duration_ms": convert_us_to_ms(metric_value(query_event, "duration", "query-duration", "query_duration")),
        "prompt_tokens": metric_value(query_event, "num-prompt-tokens", "prompt_tokens"),
        "prompt_processing_rate": metric_value(query_event, "prompt-processing-rate", "prompt_processing_rate"),
        "ttft_ms": convert_us_to_ms(metric_value(query_event, "time-to-first-token", "ttft", "ttft-us")),
        "generated_tokens": metric_value(query_event, "num-generated-tokens", "generated_tokens"),
        "token_generation_time_ms": convert_us_to_ms(
            metric_value(query_event, "token-generation-time", "token_generation_time")
        ),
        "token_generation_rate": metric_value(query_event, "token-generation-rate", "token_generation_rate"),
        "ssd_acceptance_rate": metric_value(query_event, "token-acceptance-rate", "ssd_acceptance_rate"),
    }


def input_embedding_arg(deploy_root: Path, idx: int, scale: str, offset: str) -> str:
    embed_path = deploy_root / "test_data_onboard_1664" / f"inputs_embeds_{idx}_uint16.bin"
    return f"{embed_path},uint16,{scale},{offset}"


def genie_command(
    deploy_root: Path,
    config_path: Path,
    profile_path: Path,
    idx: int,
    timeout_sec: int,
    scale: str,
    offset: str,
) -> List[str]:
    inner = (
        f"export LD_LIBRARY_PATH={deploy_root}/lib; "
        f"export ADSP_LIBRARY_PATH={deploy_root}/lib/dsp:/usr/lib/dsp/cdsp:/usr/lib/dsp/cdsp1; "
        f"cd {deploy_root}; "
        f"timeout {timeout_sec} ./bin/genie-t2t-run "
        f"-c {config_path} "
        f"-e {input_embedding_arg(deploy_root, idx, scale, offset)} "
        f"--log error "
        f"--profile {profile_path}"
    )
    return ["sg", "fastrpc", "-c", inner]


def relay_child_output(master_fd: int, stdout_path: Path, stream_response: bool) -> None:
    """Copy child PTY output to the per-run log and optionally to this terminal."""
    with stdout_path.open("wb") as stdout_handle:
        while True:
            try:
                readable, _, _ = select.select([master_fd], [], [], 0.1)
            except OSError:
                break
            if not readable:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            stdout_handle.write(chunk)
            stdout_handle.flush()
            if stream_response:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    stream_response = False


def run_one(
    deploy_root: Path,
    config_path: Path,
    paths: Dict[str, Path],
    idx: int,
    rep: Optional[int],
    timeout_sec: int,
    sample_ms: int,
    scale: str,
    offset: str,
    stream_response: bool,
) -> Dict[str, Any]:
    if rep is None:
        stem = f"warmup_idx{idx}"
    else:
        stem = f"run_idx{idx}_rep{rep}"
    profile_path = paths["runs"] / f"{stem}.json"
    stdout_path = paths["runs"] / f"{stem}.stdout.txt"
    samples_path = paths["samples"] / f"{stem}_process_samples.csv"
    profile_path.unlink(missing_ok=True)

    cmd = genie_command(deploy_root, config_path, profile_path, idx, timeout_sec, scale, offset)
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(deploy_root),
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        sampler = ProcessSampler(proc.pid, sample_ms / 1000.0, samples_path)
        sampler.start()
        if stream_response:
            label = f"warmup idx={idx}" if rep is None else f"idx={idx} rep={rep}"
            print(f"\n===== Genie response: {label} =====", flush=True)
        relay_child_output(master_fd, stdout_path, stream_response)
        returncode = proc.wait()
        sampler.stop()
    finally:
        if slave_fd != -1:
            os.close(slave_fd)
        os.close(master_fd)

    record: Dict[str, Any] = {
        "idx": idx,
        "rep": rep,
        "status": "ok" if returncode == 0 and profile_path.exists() else "failed",
        "returncode": returncode,
        "profile_json": str(profile_path),
        "stdout": str(stdout_path),
        "samples_csv": str(samples_path),
        "peak_rss_bytes": sampler.peak_rss or None,
        "peak_pss_bytes": sampler.peak_pss or None,
        "sampled_process_names": sampler.seen_names,
        "command": cmd,
        "error": "",
    }
    if profile_path.exists():
        try:
            record.update(parse_genie_profile(profile_path))
        except Exception as exc:  # pylint: disable=broad-except
            record["status"] = "failed"
            record["error"] = f"profile parse failed: {exc}"
    else:
        record["error"] = "profile JSON was not created"

    rate = record.get("token_generation_rate")
    record["regression_candidate"] = bool(rate is not None and rate < 22.0)
    return record


def numeric_values(records: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return values


def stats_for(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build_summary(records: List[Dict[str, Any]], warmup: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    per_idx: Dict[str, Dict[str, Any]] = {}
    for idx in sorted({record["idx"] for record in records}):
        idx_records = [record for record in ok_records if record.get("idx") == idx]
        per_idx[str(idx)] = {
            metric: stats_for(numeric_values(idx_records, metric))
            for metric in (
                "init_time_ms",
                "query_duration_ms",
                "prompt_processing_rate",
                "ttft_ms",
                "token_generation_time_ms",
                "token_generation_rate",
                "ssd_acceptance_rate",
                "peak_rss_bytes",
                "peak_pss_bytes",
            )
        }
        mean_rate = per_idx[str(idx)]["token_generation_rate"]["mean"]
        per_idx[str(idx)]["regression_candidate"] = bool(mean_rate is not None and mean_rate < 23.0)

    overall_mean_rate = stats_for(numeric_values(ok_records, "token_generation_rate"))["mean"]
    return {
        "warmup": warmup,
        "runs": records,
        "per_idx": per_idx,
        "overall": {
            "run_count": len(records),
            "ok_count": len(ok_records),
            "failed_count": len(records) - len(ok_records),
            "generation_rate": stats_for(numeric_values(ok_records, "token_generation_rate")),
            "ttft_ms": stats_for(numeric_values(ok_records, "ttft_ms")),
            "prompt_processing_rate": stats_for(numeric_values(ok_records, "prompt_processing_rate")),
            "peak_rss_bytes": max(numeric_values(ok_records, "peak_rss_bytes") or [0]),
            "peak_pss_bytes": max(numeric_values(ok_records, "peak_pss_bytes") or [0]),
            "regression_candidate": bool(overall_mean_rate is not None and overall_mean_rate < 23.0),
        },
    }


def write_summary_csv(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root", type=Path, default=DEPLOY_ROOT)
    parser.add_argument("--case-name", default="dual_cdsp_baseline")
    parser.add_argument("--indices", type=parse_indices, default=parse_indices(DEFAULT_INDICES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-index", type=int, default=14)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Optional source Genie config; defaults to qwen3vl.json")
    parser.add_argument("--embed-scale", default=DEFAULT_EMBED_SCALE)
    parser.add_argument("--embed-offset", default=DEFAULT_EMBED_OFFSET)
    parser.add_argument("--skip-sidecars", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--no-stream-response",
        action="store_true",
        help="Do not mirror Genie stdout/stderr to the terminal; logs are still saved under runs/.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.sample_ms < 20:
        raise SystemExit("--sample-ms must be >= 20")

    stamp, start_iso, _, _ = local_timestamp()
    output_dir = args.output_root / f"qwen3vl_{args.case_name}_{stamp}"
    paths = mkdirs(output_dir)

    snapshot_result = snapshot_configs(args.deploy_root, paths["snapshot"])
    config_path, case_info = build_case_config(args.deploy_root, args.case_name, paths["snapshot"], args.config)

    metadata = collect_metadata(args.deploy_root)
    metadata.update(
        {
            "case_name": args.case_name,
            "indices": args.indices,
            "repeats": args.repeats,
            "warmup_index": args.warmup_index,
            "sample_ms": args.sample_ms,
            "timeout_sec": args.timeout_sec,
            "output_dir": str(output_dir),
            "config_snapshot": snapshot_result,
            "case_config": case_info,
        }
    )
    write_json(metadata, output_dir / "metadata.json")

    sidecars: List[Optional[subprocess.Popen]] = []
    if not args.skip_sidecars:
        sidecars = [
            start_sidecar(["pidstat", "-u", "-r", "-d", "-h", "1"], output_dir / "pidstat.log", args.deploy_root),
            start_sidecar(["iostat", "-xz", "1"], output_dir / "iostat.log", args.deploy_root),
        ]

    warmup_record: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []
    try:
        if not args.skip_warmup:
            warmup_record = run_one(
                args.deploy_root,
                config_path,
                paths,
                args.warmup_index,
                None,
                args.timeout_sec,
                args.sample_ms,
                args.embed_scale,
                args.embed_offset,
                not args.no_stream_response,
            )

        for idx in args.indices:
            for rep in range(args.repeats):
                records.append(
                    run_one(
                        args.deploy_root,
                        config_path,
                        paths,
                        idx,
                        rep,
                        args.timeout_sec,
                        args.sample_ms,
                        args.embed_scale,
                        args.embed_offset,
                        not args.no_stream_response,
                    )
                )
                write_summary_csv(records, output_dir / "summary.csv")
                write_json(build_summary(records, warmup_record), output_dir / "summary.json")
    finally:
        for proc in sidecars:
            stop_sidecar(proc)
        if not args.skip_sidecars:
            collect_journal(output_dir / "journal_qnn.log", start_iso, args.deploy_root)

    summary = build_summary(records, warmup_record)
    write_summary_csv(records, output_dir / "summary.csv")
    write_json(summary, output_dir / "summary.json")
    print(json.dumps({"output_dir": str(output_dir), "overall": summary["overall"]}, indent=2))
    return 0 if summary["overall"]["failed_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
