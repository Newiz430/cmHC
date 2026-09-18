from __future__ import annotations

import base64
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange


STREAM_METRIC_MAX_SAMPLES = 4096


def _ensure_dir(path: Path):
    path.mkdir(parents = True, exist_ok = True)


def _get_plt():
    import matplotlib.pyplot as plt

    return plt


def build_hres_heatmap_steps(start_step: int, final_step: int, num_points: int) -> list[int]:
    if final_step < start_step:
        return [start_step]

    num_points = max(2, int(num_points))
    steps = np.linspace(start_step, final_step, num = num_points)
    steps = [int(round(step)) for step in steps.tolist()]
    steps = sorted(set(steps))

    if steps[0] != start_step:
        steps.insert(0, start_step)
    if steps[-1] != final_step:
        steps.append(final_step)

    return steps


def dobrushin_mixing_score(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype = np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0

    overlap = min(
        float(np.minimum(matrix[i], matrix[j]).sum())
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[0])
    )
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def parse_hres_module_name(module_name: str) -> tuple[int | None, str | None]:
    match = re.search(r"(?:^|\.)(?:h|blocks)\.(\d+)\.", module_name)
    layer_index = int(match.group(1)) if match else None
    if module_name.endswith("hc_attn"):
        return layer_index, "attn"
    if module_name.endswith("hc_mlp"):
        return layer_index, "mlp"
    return layer_index, None


def hres_block_index(module_name: str) -> tuple[int | None, int | None, str | None]:
    transformer_layer, kind = parse_hres_module_name(module_name)
    if transformer_layer is None or kind is None:
        return None, transformer_layer, kind
    kind_offset = 0 if kind == "attn" else 1
    return transformer_layer * 2 + kind_offset, transformer_layer, kind


def selected_stream_metric_layers(num_layers: int, window: int = 5) -> list[tuple[str, int, int]]:
    num_layers = int(num_layers) * 2
    window = int(window)
    if num_layers <= 0:
        return []

    def clipped_range(start: int):
        start = min(max(start, 0), max(num_layers - window, 0))
        return [layer for layer in range(start, min(start + window, num_layers))]

    front = clipped_range(0)
    middle = clipped_range((num_layers // 2) - (window // 2))
    back = clipped_range(num_layers - window)

    selected = []
    for group_name, layers in [("front", front), ("middle", middle), ("back", back)]:
        for slot, layer in enumerate(layers):
            selected.append((group_name, slot, layer))
    return selected


def selected_stream_metric_module_names(modules: list[tuple[str, Any]], num_layers: int) -> set[str]:
    selected_blocks = {layer for _, _, layer in selected_stream_metric_layers(num_layers)}
    if not selected_blocks:
        return set()

    selected_names = set()
    for module_name, module in modules:
        if int(getattr(module, "num_residual_streams", 0) or 0) != 4:
            continue
        block_index, _, _ = hres_block_index(module_name)
        if block_index in selected_blocks:
            selected_names.add(module_name)
    return selected_names


def init_hres_capture_state(module):
    module._supports_hres_capture = True
    module._hres_capture_enabled = False
    module._hres_stream_capture_enabled = False
    module._hres_capture_name = None
    module._hres_matrix_sum = None
    module._hres_matrix_sq_sum = None
    module._hres_spectral_sum = 0.0
    module._hres_capture_count = 0
    module._hres_coeff_sum = None
    module._hres_coeff_sq_sum = None
    module._hres_coeff_capture_count = 0
    module._hres_coeff_num_streams = None
    module._hres_coeff_init_permutation_index = None
    module._hres_coeff_fixed_identity = False
    module._hres_stream_norm_l2_sum = None
    module._hres_stream_bias_norm_l2_sum = None
    module._hres_stream_norm_fro_sum = None
    module._hres_stream_bias_norm_fro_sum = None
    module._hres_stream_cosine_sum = None
    module._hres_stream_cka_sum = None
    module._hres_stream_sample_count_sum = 0
    module._hres_stream_capture_count = 0


def set_hres_capture(module, enabled: bool, name: str | None = None):
    module._hres_capture_enabled = bool(enabled)
    if name is not None:
        module._hres_capture_name = name
    if not enabled:
        reset_hres_capture_state(module)


def set_hres_stream_capture(module, enabled: bool):
    module._hres_stream_capture_enabled = bool(enabled)
    if not enabled:
        module._hres_stream_norm_l2_sum = None
        module._hres_stream_bias_norm_l2_sum = None
        module._hres_stream_norm_fro_sum = None
        module._hres_stream_bias_norm_fro_sum = None
        module._hres_stream_cosine_sum = None
        module._hres_stream_cka_sum = None
        module._hres_stream_sample_count_sum = 0
        module._hres_stream_capture_count = 0


def reset_hres_capture_state(module):
    module._hres_matrix_sum = None
    module._hres_matrix_sq_sum = None
    module._hres_spectral_sum = 0.0
    module._hres_capture_count = 0
    module._hres_coeff_sum = None
    module._hres_coeff_sq_sum = None
    module._hres_coeff_capture_count = 0
    module._hres_coeff_num_streams = None
    module._hres_coeff_init_permutation_index = None
    module._hres_coeff_fixed_identity = False
    module._hres_stream_norm_l2_sum = None
    module._hres_stream_bias_norm_l2_sum = None
    module._hres_stream_norm_fro_sum = None
    module._hres_stream_bias_norm_fro_sum = None
    module._hres_stream_cosine_sum = None
    module._hres_stream_cka_sum = None
    module._hres_stream_sample_count_sum = 0
    module._hres_stream_capture_count = 0


def consume_hres_capture_state(module):
    if getattr(module, "_hres_capture_count", 0) == 0:
        return None

    matrix_sum = module._hres_matrix_sum
    matrix_sq_sum = module._hres_matrix_sq_sum
    spectral_sum = module._hres_spectral_sum
    capture_count = int(module._hres_capture_count)

    mean_matrix = matrix_sum / capture_count
    mean_square_matrix = matrix_sq_sum / capture_count
    std_matrix = np.sqrt(np.maximum(mean_square_matrix - np.square(mean_matrix), 0.0))
    mean_spectral = spectral_sum / capture_count

    payload = {
        "matrix": mean_matrix.astype(np.float32, copy = False),
        "matrix_std": std_matrix.astype(np.float32, copy = False),
        "matrix_sum": matrix_sum.astype(np.float32, copy = False),
        "matrix_sq_sum": matrix_sq_sum.astype(np.float32, copy = False),
        "spectral_sum": float(spectral_sum),
        "capture_count": capture_count,
        "spectral_norm": float(mean_spectral),
    }

    coefficient_count = int(getattr(module, "_hres_coeff_capture_count", 0))
    if coefficient_count > 0:
        payload["coefficient_count"] = coefficient_count
        payload["coefficient_num_streams"] = getattr(module, "_hres_coeff_num_streams", None)
        payload["coefficient_init_permutation_index"] = getattr(module, "_hres_coeff_init_permutation_index", None)
        payload["coefficient_fixed_identity"] = bool(getattr(module, "_hres_coeff_fixed_identity", False))

        coeff_sum = getattr(module, "_hres_coeff_sum", None)
        coeff_sq_sum = getattr(module, "_hres_coeff_sq_sum", None)
        if coeff_sum is not None and coeff_sq_sum is not None:
            mean_coefficients = coeff_sum / coefficient_count
            mean_square_coefficients = coeff_sq_sum / coefficient_count
            std_coefficients = np.sqrt(np.maximum(mean_square_coefficients - np.square(mean_coefficients), 0.0))
            payload["coefficients"] = mean_coefficients.astype(np.float32, copy = False)
            payload["coefficients_std"] = std_coefficients.astype(np.float32, copy = False)

    stream_count = int(getattr(module, "_hres_stream_capture_count", 0))
    if stream_count > 0:
        payload["stream_capture_count"] = stream_count
        payload["stream_sample_count"] = float(getattr(module, "_hres_stream_sample_count_sum", 0) / stream_count)
        payload["stream_norm_l2"] = (module._hres_stream_norm_l2_sum / stream_count).astype(np.float32, copy = False)
        payload["stream_norm_fro"] = (module._hres_stream_norm_fro_sum / stream_count).astype(np.float32, copy = False)
        if getattr(module, "_hres_stream_bias_norm_l2_sum", None) is not None:
            payload["stream_bias_norm_l2"] = (module._hres_stream_bias_norm_l2_sum / stream_count).astype(np.float32, copy = False)
        if getattr(module, "_hres_stream_bias_norm_fro_sum", None) is not None:
            payload["stream_bias_norm_fro"] = (module._hres_stream_bias_norm_fro_sum / stream_count).astype(np.float32, copy = False)
        payload["stream_pair_cosines"] = (module._hres_stream_cosine_sum / stream_count).astype(np.float32, copy = False)
        payload["stream_pair_linear_cka"] = (module._hres_stream_cka_sum / stream_count).astype(np.float32, copy = False)

    reset_hres_capture_state(module)
    return payload


def _linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.mean(dim = 0, keepdim = True)
    y = y - y.mean(dim = 0, keepdim = True)
    xty = x.transpose(0, 1) @ y
    xtx = x.transpose(0, 1) @ x
    yty = y.transpose(0, 1) @ y
    numerator = xty.square().sum()
    denominator = torch.linalg.matrix_norm(xtx, ord = "fro") * torch.linalg.matrix_norm(yty, ord = "fro")
    return numerator / denominator.clamp_min(eps)


def record_stream_capture_state(module, residuals: torch.Tensor):
    if not getattr(module, "_hres_stream_capture_enabled", False):
        return

    if residuals.shape[-2] != 4:
        return

    stream_vectors = residuals.detach().float()
    stream_vectors = stream_vectors.reshape(-1, stream_vectors.shape[-2], stream_vectors.shape[-1])
    if stream_vectors.shape[0] > STREAM_METRIC_MAX_SAMPLES:
        indices = torch.linspace(
            0,
            stream_vectors.shape[0] - 1,
            steps = STREAM_METRIC_MAX_SAMPLES,
            device = stream_vectors.device,
        ).long()
        stream_vectors = stream_vectors.index_select(0, indices)
    stream_bias_vectors = stream_vectors - stream_vectors.mean(dim = -2, keepdim = True)
    streams = stream_vectors.transpose(0, 1).contiguous()
    bias_streams = stream_bias_vectors.transpose(0, 1).contiguous()
    pair_indices = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    norm_l2_values = streams.norm(dim = -1).mean(dim = 1)
    bias_norm_l2_values = bias_streams.norm(dim = -1).mean(dim = 1)
    norm_fro_values = streams.flatten(1).norm(dim = -1)
    bias_norm_fro_values = bias_streams.flatten(1).norm(dim = -1)
    cosine_values = []
    cka_values = []
    for i, j in pair_indices:
        cosine_values.append(torch.nn.functional.cosine_similarity(streams[i], streams[j], dim = -1).mean())
        cka_values.append(_linear_cka(streams[i], streams[j]))

    norm_l2_values = norm_l2_values.cpu().numpy()
    bias_norm_l2_values = bias_norm_l2_values.cpu().numpy()
    norm_fro_values = norm_fro_values.cpu().numpy()
    bias_norm_fro_values = bias_norm_fro_values.cpu().numpy()
    cosine_values = torch.stack(cosine_values).cpu().numpy()
    cka_values = torch.stack(cka_values).cpu().numpy()

    if module._hres_stream_norm_l2_sum is None:
        module._hres_stream_norm_l2_sum = norm_l2_values
        module._hres_stream_bias_norm_l2_sum = bias_norm_l2_values
        module._hres_stream_norm_fro_sum = norm_fro_values
        module._hres_stream_bias_norm_fro_sum = bias_norm_fro_values
        module._hres_stream_cosine_sum = cosine_values
        module._hres_stream_cka_sum = cka_values
    else:
        module._hres_stream_norm_l2_sum += norm_l2_values
        module._hres_stream_bias_norm_l2_sum += bias_norm_l2_values
        module._hres_stream_norm_fro_sum += norm_fro_values
        module._hres_stream_bias_norm_fro_sum += bias_norm_fro_values
        module._hres_stream_cosine_sum += cosine_values
        module._hres_stream_cka_sum += cka_values

    module._hres_stream_sample_count_sum += int(stream_vectors.shape[0])
    module._hres_stream_capture_count += 1


def record_hres_capture_state(
    module,
    alpha_residual: torch.Tensor,
    *,
    permutation_coefficients: torch.Tensor | None = None,
    coefficient_num_streams: int | None = None,
    init_permutation_index: int | None = None,
    fixed_identity: bool = False,
):
    if not getattr(module, "_hres_capture_enabled", False):
        return

    matrices = rearrange(
        alpha_residual.detach().float(),
        '... f1 s1 f2 s2 -> (...) (f1 s1) (f2 s2)'
    )
    matrix_sum = matrices.sum(dim = 0).cpu().numpy()
    matrix_sq_sum = matrices.square().sum(dim = 0).cpu().numpy()
    spectral_sum = torch.linalg.matrix_norm(matrices, ord = 2).sum().item()
    capture_count = int(matrices.shape[0])

    if module._hres_matrix_sum is None:
        module._hres_matrix_sum = matrix_sum
        module._hres_matrix_sq_sum = matrix_sq_sum
    else:
        module._hres_matrix_sum += matrix_sum
        module._hres_matrix_sq_sum += matrix_sq_sum

    module._hres_spectral_sum += spectral_sum
    module._hres_capture_count += capture_count

    if coefficient_num_streams is not None:
        module._hres_coeff_num_streams = int(coefficient_num_streams)

    if init_permutation_index is not None:
        module._hres_coeff_init_permutation_index = int(init_permutation_index)

    if fixed_identity:
        module._hres_coeff_fixed_identity = True
        module._hres_coeff_capture_count += capture_count
        return

    if permutation_coefficients is None:
        return

    coefficients = permutation_coefficients.detach().float().reshape(-1, permutation_coefficients.shape[-1])
    coefficient_sum = coefficients.sum(dim = 0).cpu().numpy()
    coefficient_sq_sum = coefficients.square().sum(dim = 0).cpu().numpy()
    coefficient_count = int(coefficients.shape[0])

    if module._hres_coeff_sum is None:
        module._hres_coeff_sum = coefficient_sum
        module._hres_coeff_sq_sum = coefficient_sq_sum
    else:
        module._hres_coeff_sum += coefficient_sum
        module._hres_coeff_sq_sum += coefficient_sq_sum

    module._hres_coeff_capture_count += coefficient_count


def supports_hres_capture(module) -> bool:
    return bool(getattr(module, "_supports_hres_capture", False))


def get_hres_capture_modules(model) -> list[tuple[str, Any]]:
    return [(name, module) for name, module in model.named_modules() if supports_hres_capture(module)]


def set_hres_capture_enabled(modules: list[tuple[str, Any]], enabled: bool):
    for name, module in modules:
        module.set_hres_capture(enabled, name = name)


def set_hres_stream_capture_enabled(
    modules: list[tuple[str, Any]],
    enabled: bool,
    selected_module_names: set[str] | None = None,
):
    selected_module_names = selected_module_names or set()
    for name, module in modules:
        should_enable = bool(enabled) and name in selected_module_names
        if hasattr(module, "set_hres_stream_capture"):
            module.set_hres_stream_capture(should_enable)


def reset_hres_capture_buffers(modules: list[tuple[str, Any]]):
    for _, module in modules:
        module.reset_hres_capture_state()


def consume_hres_capture_payload(modules: list[tuple[str, Any]]):
    payload = {}
    for name, module in modules:
        module_payload = module.consume_hres_capture_state()
        if module_payload is None:
            raise RuntimeError(f"H_res capture buffer for {name} is empty")
        payload[name] = module_payload
    return payload


@torch.no_grad()
def capture_hres_probe_snapshot(model, modules, probe_batch, ctx, stream_module_names: set[str] | None = None):
    if len(modules) == 0:
        return None

    reset_hres_capture_buffers(modules)
    set_hres_capture_enabled(modules, True)
    set_hres_stream_capture_enabled(modules, stream_module_names is not None, stream_module_names)

    was_training = model.training
    model.eval()

    probe_x, probe_y = probe_batch
    with ctx:
        model(probe_x, probe_y)

    payload = consume_hres_capture_payload(modules)
    set_hres_stream_capture_enabled(modules, False)
    set_hres_capture_enabled(modules, False)

    if was_training:
        model.train()

    return payload


class CsvTrainingRecorder:
    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        _ensure_dir(self.results_dir)

    def _append_row(self, filename: str, fieldnames: list[str], row: dict[str, Any]):
        self._append_rows(filename, fieldnames, [row])

    def _append_rows(self, filename: str, fieldnames: list[str], rows: list[dict[str, Any]]):
        if not rows:
            return

        versioned_stream_files = {
            "stream_metrics.csv",
            "stream_metrics_detail.csv",
            "stream_metrics_wide.csv",
        }

        if filename in versioned_stream_files:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            candidates = []
            base_path = self.results_dir / filename
            if base_path.exists():
                candidates.append((1, base_path))
            for candidate in self.results_dir.glob(f"{stem}_v*{suffix}"):
                match = re.fullmatch(rf"{re.escape(stem)}_v(\d+){re.escape(suffix)}", candidate.name)
                if match:
                    candidates.append((int(match.group(1)), candidate))

            if not candidates:
                path = base_path
                should_write_header = True
            else:
                highest_version, highest_path = max(candidates, key = lambda item: item[0])
                if highest_path.stat().st_size == 0:
                    path = highest_path
                    should_write_header = True
                else:
                    with highest_path.open("r", encoding = "utf-8", newline = "") as csv_file:
                        existing_fieldnames = next(csv.reader(csv_file), [])
                    if existing_fieldnames == fieldnames:
                        path = highest_path
                        should_write_header = False
                    else:
                        next_version = highest_version + 1
                        path = self.results_dir / f"{stem}_v{next_version}{suffix}"
                        should_write_header = True
        else:
            path = self.results_dir / filename
            should_write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding = "utf-8", newline = "") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = fieldnames, extrasaction = "ignore")
            if should_write_header:
                writer.writeheader()
            writer.writerows(rows)

    def log_iter(self, *, step: int, train_loss: float, grad_norm: float, lr: float | None = None, throughput: float | None = None):
        self._append_row(
            "iter_metrics.csv",
            ["step", "train_loss", "grad_norm", "lr", "throughput"],
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "grad_norm": float(grad_norm),
                "lr": "" if lr is None else float(lr),
                "throughput": "" if throughput is None else float(throughput),
            },
        )

    def log_step(
        self,
        *,
        step: int,
        train_loss: float,
        val_loss: float,
        avg_train_loss: float,
        std_train_loss: float,
        avg_grad_norm: float,
        std_grad_norm: float,
        top1: float | None = None,
        top5: float | None = None,
        label: str = "eval",
    ):
        self._append_row(
            "step_metrics.csv",
            [
                "step",
                "label",
                "train_loss",
                "val_loss",
                "avg_train_loss",
                "std_train_loss",
                "avg_grad_norm",
                "std_grad_norm",
                "top1",
                "top5",
            ],
            {
                "step": int(step),
                "label": label,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "avg_train_loss": float(avg_train_loss),
                "std_train_loss": float(std_train_loss),
                "avg_grad_norm": float(avg_grad_norm),
                "std_grad_norm": float(std_grad_norm),
                "top1": "" if top1 is None else float(top1),
                "top5": "" if top5 is None else float(top5),
            },
        )

    def log_hres_snapshot(self, *, step: int, label: str, module_payloads: dict[str, dict[str, Any]] | None):
        if not module_payloads:
            return

        heatmap_fields = ["step", "label", "module_name", "row", "col", "value", "std", "dobrushin_mixing_score", "capture_count"]
        mixing_fields = ["step", "label", "module_name", "dobrushin_mixing_score", "capture_count"]
        coefficient_fields = [
            "step",
            "label",
            "module_name",
            "coefficient_index",
            "value",
            "std",
            "coefficient_count",
            "coefficient_num_streams",
            "coefficient_init_permutation_index",
            "coefficient_fixed_identity",
        ]
        heatmap_rows = []
        mixing_rows = []
        coefficient_rows = []

        for module_name, payload in module_payloads.items():
            matrix = np.asarray(payload["matrix"], dtype = np.float32)
            matrix_std = np.asarray(payload.get("matrix_std", np.zeros_like(matrix)), dtype = np.float32)
            mixing_score = dobrushin_mixing_score(matrix)
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    heatmap_rows.append({
                        "step": int(step),
                        "label": label,
                        "module_name": module_name,
                        "row": row,
                        "col": col,
                        "value": round(float(matrix[row, col]), 4),
                        "std": round(float(matrix_std[row, col]), 4),
                        "dobrushin_mixing_score": round(mixing_score, 4),
                        "capture_count": int(payload.get("capture_count", 0) or 0),
                    })

            mixing_rows.append({
                "step": int(step),
                "label": label,
                "module_name": module_name,
                "dobrushin_mixing_score": round(mixing_score, 4),
                "capture_count": int(payload.get("capture_count", 0) or 0),
            })

            coefficients = payload.get("coefficients")
            coefficient_stds = payload.get("coefficients_std")
            if coefficients is None and not bool(payload.get("coefficient_fixed_identity", False)):
                continue

            if coefficients is None:
                coefficients = np.asarray([], dtype = np.float32)
                coefficient_stds = np.asarray([], dtype = np.float32)
            else:
                coefficients = np.asarray(coefficients, dtype = np.float32)
                coefficient_stds = np.asarray(coefficient_stds if coefficient_stds is not None else np.zeros_like(coefficients), dtype = np.float32)

            if coefficients.size == 0:
                coefficient_rows.append({
                    "step": int(step),
                    "label": label,
                    "module_name": module_name,
                    "coefficient_index": "",
                    "value": "",
                    "std": "",
                    "coefficient_count": int(payload.get("coefficient_count", 0) or 0),
                    "coefficient_num_streams": payload.get("coefficient_num_streams", ""),
                    "coefficient_init_permutation_index": payload.get("coefficient_init_permutation_index", ""),
                    "coefficient_fixed_identity": bool(payload.get("coefficient_fixed_identity", False)),
                })
            for coefficient_index, coefficient_value in enumerate(coefficients):
                coefficient_rows.append({
                    "step": int(step),
                    "label": label,
                    "module_name": module_name,
                    "coefficient_index": int(coefficient_index),
                    "value": round(float(coefficient_value), 6),
                    "std": round(float(coefficient_stds[coefficient_index]), 6),
                    "coefficient_count": int(payload.get("coefficient_count", 0) or 0),
                    "coefficient_num_streams": payload.get("coefficient_num_streams", ""),
                    "coefficient_init_permutation_index": payload.get("coefficient_init_permutation_index", ""),
                    "coefficient_fixed_identity": bool(payload.get("coefficient_fixed_identity", False)),
                })

        self._append_rows("hres_heatmap_samples.csv", heatmap_fields, heatmap_rows)
        self._append_rows("hres_mixing_scores.csv", mixing_fields, mixing_rows)
        self._append_rows("hres_permutation_coefficients.csv", coefficient_fields, coefficient_rows)

    def log_stream_snapshot(
        self,
        *,
        step: int,
        label: str,
        module_payloads: dict[str, dict[str, Any]] | None,
        num_layers: int,
    ):
        if not module_payloads:
            return

        selected_layers = selected_stream_metric_layers(num_layers)
        if not selected_layers:
            return

        by_block: dict[int, list[tuple[str, int | None, str | None, dict[str, Any]]]] = {}
        for module_name, payload in module_payloads.items():
            if payload.get("stream_norm_fro") is None:
                continue
            block_index, transformer_layer, hres_kind = hres_block_index(module_name)
            if block_index is None:
                continue
            by_block.setdefault(block_index, []).append((module_name, transformer_layer, hres_kind, payload))

        wide_row: dict[str, Any] = {"step": int(step), "label": label}
        stream_rows = []
        detail_rows = []
        stream_metric_names = ["norm_l2", "norm_fro", "norm_bias_l2", "norm_bias_fro", "cosine", "linear_cka"]
        for group_name, slot, block_index in selected_layers:
            module_payload_pairs = by_block.get(block_index, [])
            if not module_payload_pairs:
                values = {metric_name: "" for metric_name in stream_metric_names}
                module_name = ""
                transformer_layer = ""
                hres_kind = ""
            else:
                module_name = module_payload_pairs[0][0]
                transformer_layer = "" if module_payload_pairs[0][1] is None else int(module_payload_pairs[0][1])
                hres_kind = "" if module_payload_pairs[0][2] is None else module_payload_pairs[0][2]
                norm_l2_values = [float(np.asarray(payload["stream_norm_l2"]).mean()) for _, _, _, payload in module_payload_pairs]
                norm_fro_values = [float(np.asarray(payload["stream_norm_fro"]).mean()) for _, _, _, payload in module_payload_pairs]
                bias_norm_l2_values = [
                    float(np.asarray(payload["stream_bias_norm_l2"]).mean())
                    for _, _, _, payload in module_payload_pairs
                    if payload.get("stream_bias_norm_l2") is not None
                ]
                bias_norm_fro_values = [
                    float(np.asarray(payload["stream_bias_norm_fro"]).mean())
                    for _, _, _, payload in module_payload_pairs
                    if payload.get("stream_bias_norm_fro") is not None
                ]
                cosine_values = [float(np.asarray(payload["stream_pair_cosines"]).mean()) for _, _, _, payload in module_payload_pairs]
                cka_values = [float(np.asarray(payload["stream_pair_linear_cka"]).mean()) for _, _, _, payload in module_payload_pairs]
                values = {
                    "norm_l2": float(np.mean(norm_l2_values)),
                    "norm_fro": float(np.mean(norm_fro_values)),
                    "norm_bias_l2": "" if not bias_norm_l2_values else float(np.mean(bias_norm_l2_values)),
                    "norm_bias_fro": "" if not bias_norm_fro_values else float(np.mean(bias_norm_fro_values)),
                    "cosine": float(np.mean(cosine_values)),
                    "linear_cka": float(np.mean(cka_values)),
                }

            for metric_name, value in values.items():
                column_name = f"{group_name}{slot}_hres{block_index}_{metric_name}"
                wide_row[column_name] = value
                stream_rows.append({
                    "step": int(step),
                    "label": label,
                    "group": group_name,
                    "slot": int(slot),
                    "layer": int(block_index),
                    "module_name": module_name,
                    "transformer_layer": transformer_layer,
                    "kind": hres_kind,
                    "metric": metric_name,
                    "value": value,
                })

            pair_names = ["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]
            for module_name, transformer_layer, hres_kind, payload in module_payload_pairs:
                block_index, _, _ = hres_block_index(module_name)
                for metric_name, key in [
                    ("norm_l2", "stream_norm_l2"),
                    ("norm_fro", "stream_norm_fro"),
                    ("norm_bias_l2", "stream_bias_norm_l2"),
                    ("norm_bias_fro", "stream_bias_norm_fro"),
                ]:
                    if payload.get(key) is None:
                        continue
                    for stream_index, value in enumerate(np.asarray(payload[key], dtype = np.float32)):
                        detail_rows.append({
                            "step": int(step),
                            "label": label,
                            "module_name": module_name,
                            "layer": block_index,
                            "transformer_layer": transformer_layer,
                            "kind": hres_kind,
                            "metric": metric_name,
                            "index": stream_index,
                            "value": float(value),
                        })
                for metric_name, key in [("cosine", "stream_pair_cosines"), ("linear_cka", "stream_pair_linear_cka")]:
                    for pair_index, value in enumerate(np.asarray(payload[key], dtype = np.float32)):
                        detail_rows.append({
                            "step": int(step),
                            "label": label,
                            "module_name": module_name,
                            "layer": block_index,
                            "transformer_layer": transformer_layer,
                            "kind": hres_kind,
                            "metric": metric_name,
                            "index": pair_names[pair_index],
                            "value": float(value),
                        })

        fieldnames = ["step", "label"] + [
            f"{group_name}{slot}_hres{block_index}_{metric_name}"
            for group_name, slot, block_index in selected_layers
            for metric_name in stream_metric_names
        ]
        self._append_rows("stream_metrics.csv", ["step", "label", "group", "slot", "layer", "module_name", "transformer_layer", "kind", "metric", "value"], stream_rows)
        self._append_rows("stream_metrics_detail.csv", ["step", "label", "module_name", "layer", "transformer_layer", "kind", "metric", "index", "value"], detail_rows)
        self._append_row("stream_metrics_wide.csv", fieldnames, wide_row)


class HResCaptureMonitor:
    def __init__(
        self,
        *,
        results_dir: str | Path,
        module_names: list[str],
        matrix_size: int,
        start_step: int,
        final_step: int,
        heatmap_points: int,
        report_title: str = "H_res Capture Report",
    ):
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.tensorboard_dir = self.results_dir / "tensorboard"
        self.plots_dir = self.results_dir / "plots"
        self.heatmap_dir = self.plots_dir / "hres_heatmaps"
        self.snapshot_dir = self.raw_dir / "heatmap_snapshots"
        self.report_path = self.results_dir / "report.html"

        for path in [self.results_dir, self.raw_dir, self.tensorboard_dir, self.plots_dir, self.heatmap_dir, self.snapshot_dir]:
            _ensure_dir(path)

        self.module_names = list(module_names)
        self.matrix_size = int(matrix_size)
        self.start_step = int(start_step)
        self.final_step = int(final_step)
        self.total_steps = max(self.final_step - self.start_step + 1, 1)
        self.heatmap_steps = build_hres_heatmap_steps(self.start_step, self.final_step, heatmap_points)
        self.recorded_rows = 0
        self.report_title = report_title
        self.snapshot_records: list[dict[str, Any]] = []
        self.has_coefficient_snapshots = False

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir = str(self.tensorboard_dir))

        self.train_loss = np.lib.format.open_memmap(
            self.raw_dir / "train_loss.npy",
            mode = "w+",
            dtype = np.float32,
            shape = (self.total_steps,),
        )
        self.grad_norm = np.lib.format.open_memmap(
            self.raw_dir / "grad_norm.npy",
            mode = "w+",
            dtype = np.float32,
            shape = (self.total_steps,),
        )
        self.hres_spectral_norm_by_module = np.lib.format.open_memmap(
            self.raw_dir / "hres_spectral_norm_by_module.npy",
            mode = "w+",
            dtype = np.float32,
            shape = (self.total_steps, len(self.module_names)),
        )
        self.hres_matrix_by_module = np.lib.format.open_memmap(
            self.raw_dir / "hres_matrix_by_module.npy",
            mode = "w+",
            dtype = np.float32,
            shape = (self.total_steps, len(self.module_names), self.matrix_size, self.matrix_size),
        )
        self.logged_step_mask = np.lib.format.open_memmap(
            self.raw_dir / "logged_step_mask.npy",
            mode = "w+",
            dtype = np.bool_,
            shape = (self.total_steps,),
        )

        self._write_metadata()

    def _write_metadata(self, *, recorded_steps: int | None = None, logged_steps: list[int] | None = None):
        payload = {
            "module_names": self.module_names,
            "matrix_size": self.matrix_size,
            "start_step": self.start_step,
            "final_step": self.final_step,
            "heatmap_steps": self.heatmap_steps,
            "recorded_steps": recorded_steps,
            "logged_steps": logged_steps,
            "report_path": str(self.report_path),
            "hres_matrix_reduction": "mean_over_batch_and_sequence_positions_per_module_per_step",
            "hres_matrix_std_reduction": "std_over_batch_and_sequence_positions_per_module_per_step",
        }
        if self.has_coefficient_snapshots:
            payload.update({
                "hres_permutation_coefficients_reduction": "mean_over_batch_and_sequence_positions_per_module_per_step",
                "hres_permutation_coefficients_std_reduction": "std_over_batch_and_sequence_positions_per_module_per_step",
            })
        (self.results_dir / "metadata.json").write_text(json.dumps(payload, indent = 2), encoding = "utf-8")

    def _row_index(self, step: int) -> int:
        row_index = int(step) - self.start_step
        if row_index < 0 or row_index >= self.total_steps:
            raise ValueError(f"step {step} is outside [{self.start_step}, {self.final_step}]")
        return row_index

    def _stack_payload(self, module_payloads: dict[str, dict[str, np.ndarray | float]]):
        matrices = np.zeros((len(self.module_names), self.matrix_size, self.matrix_size), dtype = np.float32)
        matrix_stds = np.zeros((len(self.module_names), self.matrix_size, self.matrix_size), dtype = np.float32)
        spectral_norms = np.zeros((len(self.module_names),), dtype = np.float32)

        for idx, module_name in enumerate(self.module_names):
            payload = module_payloads[module_name]
            matrices[idx] = payload["matrix"]
            matrix_stds[idx] = payload.get("matrix_std", 0.0)
            spectral_norms[idx] = payload["spectral_norm"]

        return matrices, matrix_stds, spectral_norms

    def log_training_step(
        self,
        *,
        step: int,
        train_loss: float,
        grad_norm: float,
        module_payloads: dict[str, dict[str, np.ndarray | float]],
    ):
        row_index = self._row_index(step)
        matrices, _, spectral_norms = self._stack_payload(module_payloads)

        self.train_loss[row_index] = train_loss
        self.grad_norm[row_index] = grad_norm
        self.hres_spectral_norm_by_module[row_index] = spectral_norms
        self.hres_matrix_by_module[row_index] = matrices
        if not bool(self.logged_step_mask[row_index]):
            self.recorded_rows += 1
        self.logged_step_mask[row_index] = True

        self.writer.add_scalar("train/loss", train_loss, step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)
        self.writer.add_scalar("hres/spectral_norm_mean", float(spectral_norms.mean()), step)
        self.writer.add_scalar("hres/spectral_norm_max", float(spectral_norms.max()), step)

    def capture_heatmap_snapshot(
        self,
        *,
        step: int,
        module_payloads: dict[str, dict[str, np.ndarray | float]],
        label: str,
    ):
        plt = _get_plt()
        matrices, matrix_stds, _ = self._stack_payload(module_payloads)
        mixing_scores = np.asarray([dobrushin_mixing_score(matrix) for matrix in matrices], dtype = np.float32)
        snapshot_path = self.snapshot_dir / f"{label}_step_{step:07d}.npy"
        np.save(snapshot_path, matrices)

        figure = self._build_heatmap_figure(matrices, matrix_stds, mixing_scores = mixing_scores, title = f"{label} | step {step}")
        output_path = self.heatmap_dir / f"{label}_step_{step:07d}.png"
        figure.savefig(output_path, dpi = 200)
        self.writer.add_figure(f"hres_heatmaps/{label}", figure, global_step = step)
        plt.close(figure)

        snapshot_record = {
            "label": label,
            "step": int(step),
            "matrix_path": str(snapshot_path),
            "image_path": str(output_path),
            "dobrushin_mixing_scores": mixing_scores.round(4).tolist(),
        }

        if self._has_coefficient_payload(module_payloads):
            coefficient_snapshot_path = self.snapshot_dir / f"{label}_step_{step:07d}_permutation_coefficients.json"
            coefficient_snapshot_path.write_text(
                json.dumps(self._coefficient_snapshot_payload(module_payloads), indent = 2),
                encoding = "utf-8",
            )
            coefficient_figure = self._build_coefficient_figure(
                module_payloads,
                title = f"{label} permutation coefficients | step {step}",
            )
            coefficient_output_path = self.heatmap_dir / f"{label}_step_{step:07d}_permutation_coefficients.png"
            coefficient_figure.savefig(coefficient_output_path, dpi = 200)
            self.writer.add_figure(f"hres_permutation_coefficients/{label}", coefficient_figure, global_step = step)
            plt.close(coefficient_figure)

            snapshot_record["coefficient_path"] = str(coefficient_snapshot_path)
            snapshot_record["coefficient_image_path"] = str(coefficient_output_path)
            self.has_coefficient_snapshots = True

        self.snapshot_records.append(snapshot_record)

    def _build_heatmap_figure(self, matrices: np.ndarray, matrix_stds: np.ndarray, *, mixing_scores: np.ndarray | None = None, title: str):
        plt = _get_plt()
        num_modules = len(self.module_names)
        num_cols = min(4, max(1, num_modules))
        num_rows = math.ceil(num_modules / num_cols)

        vmin = float(matrices.min())
        vmax = float(matrices.max())

        figure, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize = (4 * num_cols, 4 * num_rows),
            constrained_layout = True,
        )
        axes = np.array(axes, dtype = object).reshape(num_rows, num_cols)
        figure.suptitle(title, fontsize = 14)

        image = None
        for idx, module_name in enumerate(self.module_names):
            row = idx // num_cols
            col = idx % num_cols
            axis = axes[row, col]
            image = axis.imshow(matrices[idx], cmap = "coolwarm", vmin = vmin, vmax = vmax)
            score_text = ""
            if mixing_scores is not None:
                score_text = f"\ns_mix={float(mixing_scores[idx]):.4f}"
            axis.set_title(f"{module_name}{score_text}", fontsize = 9)
            axis.set_xticks(range(self.matrix_size))
            axis.set_yticks(range(self.matrix_size))

            threshold = (vmin + vmax) / 2 if vmax != vmin else vmin
            for matrix_row in range(self.matrix_size):
                for matrix_col in range(self.matrix_size):
                    value = float(matrices[idx, matrix_row, matrix_col])
                    std = float(matrix_stds[idx, matrix_row, matrix_col])
                    text_color = "white" if value < threshold else "#111827"
                    axis.text(
                        matrix_col,
                        matrix_row,
                        f"{value:.4f}\n±{std:.4f}",
                        ha = "center",
                        va = "center",
                        multialignment = "center",
                        linespacing = 1.05,
                        fontsize = 6.5,
                        color = text_color,
                    )

        for idx in range(num_modules, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis("off")

        if image is not None:
            figure.colorbar(image, ax = axes.ravel().tolist(), shrink = 0.8)
        return figure

    def _has_coefficient_payload(self, module_payloads: dict[str, dict[str, Any]]) -> bool:
        for module_name in self.module_names:
            payload = module_payloads[module_name]
            if payload.get("coefficients") is not None:
                return True
            if int(payload.get("coefficient_count", 0) or 0) > 0:
                return True
        return False

    def _coefficient_snapshot_payload(self, module_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
        modules = []
        for module_name in self.module_names:
            payload = module_payloads[module_name]
            coefficients = payload.get("coefficients")
            coefficient_stds = payload.get("coefficients_std")
            modules.append({
                "module_name": module_name,
                "coefficient_num_streams": payload.get("coefficient_num_streams"),
                "coefficient_count": payload.get("coefficient_count", 0),
                "coefficient_init_permutation_index": payload.get("coefficient_init_permutation_index"),
                "coefficient_fixed_identity": bool(payload.get("coefficient_fixed_identity", False)),
                "coefficients": None if coefficients is None else np.asarray(coefficients).tolist(),
                "coefficients_std": None if coefficient_stds is None else np.asarray(coefficient_stds).tolist(),
            })
        return {"module_names": self.module_names, "modules": modules}

    def _build_coefficient_figure(self, module_payloads: dict[str, dict[str, Any]], *, title: str):
        plt = _get_plt()
        num_modules = len(self.module_names)
        num_cols = min(4, max(1, num_modules))
        num_rows = math.ceil(num_modules / num_cols)

        figure, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize = (4 * num_cols, 1.35 * num_rows),
            constrained_layout = True,
        )
        axes = np.array(axes, dtype = object).reshape(num_rows, num_cols)
        figure.suptitle(title, fontsize = 12)

        for idx, module_name in enumerate(self.module_names):
            row = idx // num_cols
            col = idx % num_cols
            axis = axes[row, col]
            payload = module_payloads[module_name]
            coefficients = payload.get("coefficients")
            num_streams = payload.get("coefficient_num_streams") or self.matrix_size
            fixed_identity = bool(payload.get("coefficient_fixed_identity", False))
            init_permutation_index = payload.get("coefficient_init_permutation_index")

            axis.set_title(module_name, fontsize = 8.5, pad = 2)
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(-0.015, 0.20)
            axis.set_yticks([])
            axis.set_xticks([0.0, 0.5, 1.0])
            axis.grid(axis = "x", alpha = 0.18)
            for side in ["left", "right", "top"]:
                axis.spines[side].set_visible(False)
            axis.spines["bottom"].set_linewidth(1.0)
            axis.tick_params(axis = "x", labelsize = 7, pad = 1)
            if row == num_rows - 1:
                axis.set_xlabel("coefficient $a_i$", fontsize = 8, labelpad = 1)
            else:
                axis.tick_params(axis = "x", labelbottom = False)

            try:
                num_streams = int(num_streams)
            except (TypeError, ValueError):
                num_streams = self.matrix_size
            try:
                init_permutation_index = int(init_permutation_index)
            except (TypeError, ValueError):
                init_permutation_index = 0

            if fixed_identity:
                axis.text(
                    0.5,
                    0.085,
                    "fixed identity",
                    ha = "center",
                    va = "center",
                    fontsize = 8,
                    color = "#92400e",
                )
                continue
            elif coefficients is not None:
                coefficient_values = np.asarray(coefficients, dtype = np.float32)
                rng = np.random.default_rng(1729 + idx * 7919 + coefficient_values.size)
                y_jitter = rng.uniform(0.015, 0.12, size = coefficient_values.size)
                point_size = 18 if coefficient_values.size <= 64 else 7
                axis.scatter(coefficient_values, y_jitter, s = point_size, c = "#6b7280", alpha = 0.58, linewidths = 0)
                text_size = 5.5 if coefficient_values.size <= 128 else 3.3
                for coefficient_index, (coefficient_value, jitter_value) in enumerate(zip(coefficient_values, y_jitter)):
                    axis.text(
                        float(coefficient_value),
                        float(jitter_value) + 0.011,
                        str(coefficient_index),
                        ha = "center",
                        va = "bottom",
                        fontsize = text_size,
                        color = "#374151",
                        alpha = 0.78,
                        clip_on = True,
                        zorder = 3,
                    )
                if not 0 <= init_permutation_index < coefficient_values.size:
                    init_permutation_index = 0
                star_index = init_permutation_index
                star_a = float(coefficient_values[star_index]) if coefficient_values.size > 0 else 0.0
                star_y = 0.135
            else:
                axis.text(
                    0.5,
                    0.085,
                    "no permutation coefficients",
                    ha = "center",
                    va = "center",
                    fontsize = 8,
                    color = "#6b7280",
                )
                continue

            axis.scatter(
                [star_a],
                [star_y],
                s = 95,
                marker = "*",
                c = "#d97706",
                edgecolors = "#111827",
                linewidths = 0.8,
                zorder = 4,
            )
            if fixed_identity:
                axis.text(
                    star_a,
                    star_y + 0.026,
                    str(star_index),
                    ha = "center",
                    va = "bottom",
                    fontsize = 7.5,
                    color = "#92400e",
                    clip_on = True,
                    zorder = 5,
                )

        for idx in range(num_modules, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis("off")

        return figure

    def finalize(self):
        if self.recorded_rows == 0:
            self._write_metadata(recorded_steps = 0, logged_steps = [])
            self._close_memmaps()
            self.writer.close()
            self._write_html_report(None)
            return

        logged_mask = np.asarray(self.logged_step_mask[:self.total_steps], dtype = bool)
        steps = np.arange(self.start_step, self.final_step + 1, dtype = np.int64)[logged_mask]
        train_loss = np.asarray(self.train_loss[:self.total_steps])[logged_mask]
        grad_norm = np.asarray(self.grad_norm[:self.total_steps])[logged_mask]
        spectral_norms = np.asarray(self.hres_spectral_norm_by_module[:self.total_steps])[logged_mask]
        spectral_mean = spectral_norms.mean(axis = 1)
        spectral_max = spectral_norms.max(axis = 1)

        self._plot_curve(
            steps,
            train_loss,
            title = "Train Loss",
            ylabel = "Loss",
            output_path = self.plots_dir / "train_loss_curve.png",
        )
        self._plot_curve(
            steps,
            grad_norm,
            title = "Gradient Norm",
            ylabel = "Gradient Norm",
            output_path = self.plots_dir / "grad_norm_curve.png",
        )
        self._plot_dual_curve(
            steps,
            spectral_mean,
            spectral_max,
            title = "H_res Spectral Norm",
            ylabel = "Spectral Norm",
            output_path = self.plots_dir / "hres_spectral_norm_curve.png",
            left_label = "mean across modules",
            right_label = "max across modules",
        )

        self._write_metadata(
            recorded_steps = int(self.recorded_rows),
            logged_steps = steps.tolist(),
        )
        report_payload = {
            "steps": steps.tolist(),
            "train_loss": train_loss.tolist(),
            "grad_norm": grad_norm.tolist(),
            "spectral_norm_mean": spectral_mean.tolist(),
            "spectral_norm_max": spectral_max.tolist(),
        }
        self._close_memmaps()
        self.writer.close()
        self._write_html_report(report_payload)

    def _close_memmaps(self):
        for name in [
            "train_loss",
            "grad_norm",
            "hres_spectral_norm_by_module",
            "hres_matrix_by_module",
            "logged_step_mask",
        ]:
            array = getattr(self, name, None)
            if array is None:
                continue

            array.flush()
            mmap_obj = getattr(array, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            setattr(self, name, None)

    def _plot_curve(self, steps: np.ndarray, values: np.ndarray, *, title: str, ylabel: str, output_path: Path):
        plt = _get_plt()
        figure, axis = plt.subplots(figsize = (8, 4.5))
        axis.plot(steps, values, linewidth = 1.5)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha = 0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi = 200)
        plt.close(figure)

    def _plot_dual_curve(
        self,
        steps: np.ndarray,
        left_values: np.ndarray,
        right_values: np.ndarray,
        *,
        title: str,
        ylabel: str,
        output_path: Path,
        left_label: str,
        right_label: str,
    ):
        plt = _get_plt()
        figure, axis = plt.subplots(figsize = (8, 4.5))
        axis.plot(steps, left_values, linewidth = 1.5, label = left_label)
        axis.plot(steps, right_values, linewidth = 1.2, linestyle = "--", label = right_label)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha = 0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi = 200)
        plt.close(figure)

    def _encode_image(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _snapshot_card_html(self, snapshot: dict[str, Any]) -> str:
        image_b64 = self._encode_image(Path(snapshot["image_path"]))
        coefficient_image_html = ""
        coefficient_image_path = snapshot.get("coefficient_image_path")
        if coefficient_image_path:
            coefficient_image_b64 = self._encode_image(Path(coefficient_image_path))
            coefficient_image_html = (
                "<h4>Permutation Coefficient Distributions</h4>"
                f"<img class=\"report-image compact-report-image\" src=\"data:image/png;base64,{coefficient_image_b64}\" "
                f"alt=\"{html.escape(snapshot['label'])} permutation coefficients\" />"
            )
        panel_id = f"snapshot-panel-{snapshot['label']}-{snapshot['step']}"
        return (
            f"<section class=\"card snapshot-card snapshot-panel\" id=\"{html.escape(panel_id)}\">"
            f"<h3>{html.escape(snapshot['label'])} | step {snapshot['step']}</h3>"
            f"<img class=\"report-image\" src=\"data:image/png;base64,{image_b64}\" alt=\"{html.escape(snapshot['label'])} heatmap\" />"
            f"{coefficient_image_html}"
            "</section>"
        )

    def _write_html_report(self, report_payload: dict[str, Any] | None):
        metadata = json.loads((self.results_dir / "metadata.json").read_text(encoding = "utf-8"))
        metadata_for_html = {k: v for k, v in metadata.items() if k != "logged_steps"}
        report_payload = report_payload or {}
        coefficient_text = (
            "The permutation coefficient plot below each heatmap uses the same module layout; "
            "gray points are mean coefficients `a_i`, each point is annotated by its permutation index, "
            "and the orange star marks the initialization permutation index."
            if self.has_coefficient_snapshots
            else "No permutation coefficient plots were recorded for this run."
        )
        hero_items = "scalar curves, sampled heatmaps, metadata, and embedded snapshot matrices"
        if self.has_coefficient_snapshots:
            hero_items = "scalar curves, sampled heatmaps, permutation coefficient distributions, metadata, and embedded snapshot matrices"

        cards = [
            ("Recorded steps", str(metadata.get("recorded_steps", 0) or 0)),
            ("Modules", str(len(self.module_names))),
            ("Matrix size", f"{self.matrix_size} x {self.matrix_size}"),
            ("Heatmap samples", str(len(self.snapshot_records))),
        ]
        summary_cards = "".join(
            f"<div class=\"metric-card\"><div class=\"metric-label\">{html.escape(label)}</div><div class=\"metric-value\">{html.escape(value)}</div></div>"
            for label, value in cards
        )

        curve_sections = []
        for title, filename in [
            ("Train Loss", "train_loss_curve.png"),
            ("Gradient Norm", "grad_norm_curve.png"),
            ("H_res Spectral Norm", "hres_spectral_norm_curve.png"),
        ]:
            image_path = self.plots_dir / filename
            if image_path.exists():
                curve_sections.append(
                    "<section class=\"card\">"
                    f"<h3>{html.escape(title)}</h3>"
                    f"<img class=\"report-image\" src=\"data:image/png;base64,{self._encode_image(image_path)}\" alt=\"{html.escape(title)}\" />"
                    "</section>"
                )

        snapshot_tabs = []
        snapshot_panels = []
        for index, snapshot in enumerate(self.snapshot_records):
            panel_id = f"snapshot-panel-{snapshot['label']}-{snapshot['step']}"
            active_class = " active" if index == 0 else ""
            snapshot_tabs.append(
                f"<button class=\"snapshot-tab-button{active_class}\" type=\"button\" data-target=\"{html.escape(panel_id)}\">"
                f"{html.escape(snapshot['label'])} | step {snapshot['step']}"
                "</button>"
            )
            snapshot_panels.append(self._snapshot_card_html(snapshot))

        snapshot_section_html = (
            "<div class=\"snapshot-tabs\">"
            f"{''.join(snapshot_tabs)}"
            "</div>"
            "<div class=\"snapshot-panels\">"
            f"{''.join(snapshot_panels)}"
            "</div>"
        ) if self.snapshot_records else '<p>No heatmap snapshots were recorded.</p>'

        html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(self.report_title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #16202a;
      --muted: #5d6a78;
      --accent: #0b6e99;
      --border: #d9e1ea;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: linear-gradient(180deg, #f7fafc 0%, #eef3f8 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      padding: 24px 28px;
      border-radius: 20px;
      background: linear-gradient(135deg, #0b6e99 0%, #7cc6e5 100%);
      color: white;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .hero p {{ margin: 8px 0 0; opacity: 0.92; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin: 24px 0;
    }}
    .metric-card, .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{
      padding: 18px 20px;
    }}
    .metric-label {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 26px;
      font-weight: 700;
    }}
    .card {{
      padding: 20px;
      margin-bottom: 20px;
    }}
    .report-image {{
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: white;
    }}
    .compact-report-image {{
      margin-top: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .snapshot-tabs {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 18px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    .snapshot-tab-button {{
      border: 1px solid var(--border);
      border-bottom: none;
      background: linear-gradient(180deg, #f5f8fb 0%, #e9eff6 100%);
      color: var(--muted);
      padding: 10px 16px;
      border-radius: 14px 14px 0 0;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
    }}
    .snapshot-tab-button:hover {{
      transform: translateY(-1px);
      color: var(--text);
    }}
    .snapshot-tab-button.active {{
      background: linear-gradient(180deg, #0b6e99 0%, #0f8cc2 100%);
      color: white;
      border-color: #0b6e99;
      box-shadow: 0 8px 20px rgba(11, 110, 153, 0.18);
    }}
    .snapshot-panels {{
      position: relative;
    }}
    .snapshot-panel {{
      display: none;
      margin-bottom: 0;
    }}
    .snapshot-panel.active {{
      display: block;
    }}
    pre {{
      overflow-x: auto;
      background: #0f172a;
      color: #e2e8f0;
      padding: 16px;
      border-radius: 12px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(self.report_title)}</h1>
      <p>Open this file directly to inspect {html.escape(hero_items)}.</p>
    </section>

    <section class="metrics">
      {summary_cards}
    </section>

    <section class="card">
      <h2>Run Metadata</h2>
      <pre>{html.escape(json.dumps(metadata_for_html, indent = 2, ensure_ascii = False))}</pre>
    </section>

    <section class="card">
      <h2>Heatmap Labels</h2>
      <p>`transformer.h.&lt;l&gt;.hc_attn` is the H_res matrix for the attention branch in block `l`, and `transformer.h.&lt;l&gt;.hc_mlp` is the H_res matrix for the MLP branch in the same block. Each heatmap cell shows the mean on the first line and `+/-` standard deviation on the second line, both reduced over batch and sequence positions. {html.escape(coefficient_text)}</p>
    </section>

    <section class="grid">
      {''.join(curve_sections) if curve_sections else '<section class="card"><h3>No scalar curves recorded</h3></section>'}
    </section>

    <section class="card">
      <h2>Heatmap Snapshots</h2>
      {snapshot_section_html}
    </section>
  </div>
  <script>
    (() => {{
      const buttons = Array.from(document.querySelectorAll('.snapshot-tab-button'));
      const panels = Array.from(document.querySelectorAll('.snapshot-panel'));
      if (buttons.length === 0 || panels.length === 0) {{
        return;
      }}

      const activate = (targetId) => {{
        for (const button of buttons) {{
          button.classList.toggle('active', button.dataset.target === targetId);
        }}
        for (const panel of panels) {{
          panel.classList.toggle('active', panel.id === targetId);
        }}
      }};

      for (const button of buttons) {{
        button.addEventListener('click', () => activate(button.dataset.target));
      }}

      activate(buttons[0].dataset.target);
    }})();
  </script>
</body>
</html>
"""
        self.report_path.write_text(html_text, encoding = "utf-8")
