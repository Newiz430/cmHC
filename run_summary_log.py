from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


RUN_SUMMARY_LOG_PATH = Path(__file__).resolve().parent / "run_summary.log"


_RUN_LOG_HEADERS_WRITTEN: set[tuple[Path, str]] = set()
_STEP_LOG_HEADERS_WRITTEN: set[tuple[Path, str]] = set()
_RUN_LOG_CONFIGS_WRITTEN: set[tuple[Path, str]] = set()
_STEP_LOG_CONFIGS_WRITTEN: set[tuple[Path, str]] = set()


def get_current_command() -> str:
    proc_cmdline_path = Path("/proc/self/cmdline")
    if proc_cmdline_path.exists():
        raw = proc_cmdline_path.read_text(encoding = "utf-8", errors = "ignore")
        command = raw.replace("\x00", " ").strip()
        if command:
            return command

    return " ".join([sys.executable, *sys.argv]).strip()


def format_eval_summary(step: int, losses, train_losses, grad_norms, phase: str | None = None) -> str:
    avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
    avg_grad_norm = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]

    prefix = f"[step {step}]"
    if phase:
        prefix += f" {phase}, "

    return prefix + ", ".join([
        f"train loss: {losses['train']:.4f}",
        f"val loss: {losses['val']:.4f}",
        f"(avg train loss: {avg_train_loss[0]:.4f} +/- {avg_train_loss[1]:.4f})",
        f"(avg grad norm: {avg_grad_norm[0]:.4f} +/- {avg_grad_norm[1]:.4f})",
    ])


def step_log_path_for(log_path: str | Path = RUN_SUMMARY_LOG_PATH) -> Path:
    path = Path(log_path)
    if path.suffix:
        return path.with_name(f"{path.stem}_step{path.suffix}")
    return path.with_name(f"{path.name}_step.log")


def _log_header_key(path: Path, run_start_timestamp: str) -> tuple[Path, str]:
    return path.absolute(), run_start_timestamp


def _write_log_header(path: Path, run_start_timestamp: str):
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("a", encoding = "utf-8") as log_file:
        log_file.write(f"{run_start_timestamp} | command: {get_current_command()}\n")


def _write_log_config(path: Path, run_start_timestamp: str, config_line: str):
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("a", encoding = "utf-8") as log_file:
        if config_line:
            log_file.write(f"{run_start_timestamp} | config: {config_line}\n")


def append_run_header(
    run_start_timestamp: str,
    log_path: str | Path | None = RUN_SUMMARY_LOG_PATH,
    config_line: str | None = None,
):
    log_path = RUN_SUMMARY_LOG_PATH if log_path is None else Path(log_path)
    step_log_path = step_log_path_for(log_path)

    run_key = _log_header_key(log_path, run_start_timestamp)
    if run_key not in _RUN_LOG_HEADERS_WRITTEN:
        _write_log_header(log_path, run_start_timestamp)
        _RUN_LOG_HEADERS_WRITTEN.add(run_key)
    if config_line and run_key not in _RUN_LOG_CONFIGS_WRITTEN:
        _write_log_config(log_path, run_start_timestamp, config_line)
        _RUN_LOG_CONFIGS_WRITTEN.add(run_key)

    step_key = _log_header_key(step_log_path, run_start_timestamp)
    if step_key not in _STEP_LOG_HEADERS_WRITTEN:
        _write_log_header(step_log_path, run_start_timestamp)
        _STEP_LOG_HEADERS_WRITTEN.add(step_key)
    if config_line and step_key not in _STEP_LOG_CONFIGS_WRITTEN:
        _write_log_config(step_log_path, run_start_timestamp, config_line)
        _STEP_LOG_CONFIGS_WRITTEN.add(step_key)


def append_run_summary(run_start_timestamp: str, summary_line: str, log_path: str | Path | None = RUN_SUMMARY_LOG_PATH):
    command = get_current_command()
    log_path = RUN_SUMMARY_LOG_PATH if log_path is None else Path(log_path)
    log_path.parent.mkdir(parents = True, exist_ok = True)

    with log_path.open("a", encoding = "utf-8") as log_file:
        if _log_header_key(log_path, run_start_timestamp) not in _RUN_LOG_HEADERS_WRITTEN:
            log_file.write(f"{run_start_timestamp} | command: {command}\n")
            _RUN_LOG_HEADERS_WRITTEN.add(_log_header_key(log_path, run_start_timestamp))
        log_file.write(f"{summary_line}\n")


def append_step_summary(run_start_timestamp: str, summary_line: str, log_path: str | Path | None = RUN_SUMMARY_LOG_PATH):
    if log_path is None:
        log_path = RUN_SUMMARY_LOG_PATH

    command = get_current_command()
    step_log_path = step_log_path_for(log_path)
    step_log_path.parent.mkdir(parents = True, exist_ok = True)

    with step_log_path.open("a", encoding = "utf-8") as log_file:
        if _log_header_key(step_log_path, run_start_timestamp) not in _STEP_LOG_HEADERS_WRITTEN:
            log_file.write(f"{run_start_timestamp} | command: {command}\n")
            _STEP_LOG_HEADERS_WRITTEN.add(_log_header_key(step_log_path, run_start_timestamp))
        log_file.write(f"{summary_line.rstrip()}\n")
