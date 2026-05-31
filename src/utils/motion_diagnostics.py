from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from utils.video import (
    get_video_info,
    iter_aligned_frames,
    make_timestamps,
)


@dataclass(frozen=True)
class MotionDiagnostics:
    image_path: Path
    target_motion_mean: float
    reference_motion_mean: float
    missing_motion_ratio: float
    extra_motion_ratio: float
    motion_direction_error: float
    strongest_missing_regions: tuple[str, ...]
    flow_pairs: int


def build_motion_diagnostics(
    target_path: str | Path,
    gt_path: str | Path,
    *,
    start: float,
    duration: float | None,
    fps: float,
    resize: str,
    output_path: str | Path,
    max_width: int,
) -> MotionDiagnostics:
    target_info = get_video_info(target_path)
    gt_info = get_video_info(gt_path)
    timestamps = make_timestamps(
        target_info,
        gt_info,
        start=start,
        duration=duration,
        fps=fps,
    )
    if len(timestamps) < 2:
        raise ValueError("motion diagnostics need at least two sampled frames")

    frames = list(iter_aligned_frames(target_path, gt_path, timestamps, resize=resize))
    return compute_motion_diagnostics_from_frames(
        frames,
        output_path=output_path,
        max_width=max_width,
    )


def compute_motion_diagnostics_from_frames(
    frames: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    output_path: str | Path,
    max_width: int,
) -> MotionDiagnostics:
    if len(frames) < 2:
        raise ValueError("motion diagnostics need at least two sampled frames")

    first_target = frames[0][1]
    first_gt = frames[0][2]
    height, width = first_target.shape[:2]
    zeros = np.zeros((height, width), dtype=np.float32)
    target_motion_sum = zeros.copy()
    gt_motion_sum = zeros.copy()
    missing_motion_sum = zeros.copy()
    extra_motion_sum = zeros.copy()
    direction_error_sum = 0.0
    direction_error_count = 0
    flow_pairs = 0

    prev_target_gray = to_gray(first_target)
    prev_gt_gray = to_gray(first_gt)

    for _timestamp, target_frame, gt_frame in frames[1:]:
        target_gray = to_gray(target_frame)
        gt_gray = to_gray(gt_frame)
        target_flow = optical_flow(prev_target_gray, target_gray)
        gt_flow = optical_flow(prev_gt_gray, gt_gray)
        target_mag = flow_magnitude(target_flow)
        gt_mag = flow_magnitude(gt_flow)

        target_motion_sum += target_mag
        gt_motion_sum += gt_mag
        missing_motion_sum += np.maximum(gt_mag - target_mag, 0.0)
        extra_motion_sum += np.maximum(target_mag - gt_mag, 0.0)

        error, count = direction_error(target_flow, gt_flow, target_mag, gt_mag)
        direction_error_sum += error
        direction_error_count += count

        prev_target_gray = target_gray
        prev_gt_gray = gt_gray
        flow_pairs += 1

    if flow_pairs == 0:
        raise ValueError("no optical-flow frame pairs were computed")

    target_motion = target_motion_sum / flow_pairs
    gt_motion = gt_motion_sum / flow_pairs
    missing_motion = missing_motion_sum / flow_pairs
    extra_motion = extra_motion_sum / flow_pairs

    reference_motion_mean = float(np.mean(gt_motion))
    target_motion_mean = float(np.mean(target_motion))
    denom = max(reference_motion_mean, 1e-6)
    missing_motion_ratio = float(np.mean(missing_motion) / denom)
    extra_motion_ratio = float(np.mean(extra_motion) / denom)
    motion_direction_error = (
        float(direction_error_sum / direction_error_count)
        if direction_error_count
        else 0.0
    )
    strongest_missing_regions = strongest_regions(missing_motion)

    image_path = Path(output_path).expanduser()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    render_motion_diagnostics_image(
        image_path,
        first_target,
        first_gt,
        target_motion,
        gt_motion,
        missing_motion,
        extra_motion,
        max_width=max_width,
    )

    return MotionDiagnostics(
        image_path=image_path,
        target_motion_mean=target_motion_mean,
        reference_motion_mean=reference_motion_mean,
        missing_motion_ratio=missing_motion_ratio,
        extra_motion_ratio=extra_motion_ratio,
        motion_direction_error=motion_direction_error,
        strongest_missing_regions=strongest_missing_regions,
        flow_pairs=flow_pairs,
    )


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def optical_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


def direction_error(
    target_flow: np.ndarray,
    gt_flow: np.ndarray,
    target_mag: np.ndarray,
    gt_mag: np.ndarray,
) -> tuple[float, int]:
    threshold = max(0.2, float(np.percentile(gt_mag, 75)) * 0.25)
    mask = (gt_mag > threshold) & (target_mag > threshold)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0, 0

    target_vec = target_flow[mask]
    gt_vec = gt_flow[mask]
    dot = np.sum(target_vec * gt_vec, axis=1)
    denom = np.linalg.norm(target_vec, axis=1) * np.linalg.norm(gt_vec, axis=1)
    cosine = np.clip(dot / np.maximum(denom, 1e-6), -1.0, 1.0)
    # 0 means same direction, 1 means opposite direction.
    return float(np.sum(0.5 * (1.0 - cosine))), count


def render_motion_diagnostics_image(
    path: Path,
    first_target: np.ndarray,
    first_gt: np.ndarray,
    target_motion: np.ndarray,
    gt_motion: np.ndarray,
    missing_motion: np.ndarray,
    extra_motion: np.ndarray,
    *,
    max_width: int,
) -> None:
    motion_scale = max(
        percentile_scale(target_motion),
        percentile_scale(gt_motion),
    )
    diff_scale = max(
        percentile_scale(missing_motion),
        percentile_scale(extra_motion),
    )

    panels = [
        make_panel(first_target, target_motion, motion_scale, "target avg motion"),
        make_panel(first_gt, gt_motion, motion_scale, "reference avg motion"),
        make_panel(first_gt, missing_motion, diff_scale, "missing motion in target"),
        make_panel(first_target, extra_motion, diff_scale, "extra motion in target"),
    ]
    gap = 8
    panel_h, panel_w = panels[0].shape[:2]
    grid = np.full((panel_h * 2 + gap, panel_w * 2 + gap, 3), 245, dtype=np.uint8)
    grid[0:panel_h, 0:panel_w] = panels[0]
    grid[0:panel_h, panel_w + gap : panel_w * 2 + gap] = panels[1]
    grid[panel_h + gap : panel_h * 2 + gap, 0:panel_w] = panels[2]
    grid[
        panel_h + gap : panel_h * 2 + gap,
        panel_w + gap : panel_w * 2 + gap,
    ] = panels[3]

    if max_width > 0 and grid.shape[1] > max_width:
        scale = max_width / grid.shape[1]
        grid = cv2.resize(
            grid,
            (max(1, int(grid.shape[1] * scale)), max(1, int(grid.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    if not cv2.imwrite(str(path), grid):
        raise RuntimeError(f"could not write motion diagnostics image: {path}")


def strongest_regions(values: np.ndarray, top_k: int = 3) -> tuple[str, ...]:
    if values.size == 0 or float(np.max(values)) <= 1e-6:
        return ()

    row_names = ("top", "middle", "bottom")
    col_names = ("left", "center", "right")
    height, width = values.shape[:2]
    row_edges = np.linspace(0, height, num=4, dtype=int)
    col_edges = np.linspace(0, width, num=4, dtype=int)
    regions: list[tuple[float, str]] = []
    for row_index, row_name in enumerate(row_names):
        for col_index, col_name in enumerate(col_names):
            y1, y2 = row_edges[row_index], row_edges[row_index + 1]
            x1, x2 = col_edges[col_index], col_edges[col_index + 1]
            score = float(np.mean(values[y1:y2, x1:x2]))
            regions.append((score, f"{row_name}-{col_name}"))

    max_score = max(score for score, _name in regions)
    threshold = max_score * 0.25
    strongest = [
        name
        for score, name in sorted(regions, key=lambda item: item[0], reverse=True)
        if score >= threshold and score > 1e-6
    ]
    return tuple(strongest[:top_k])


def percentile_scale(values: np.ndarray) -> float:
    return max(float(np.percentile(values, 95)), 1e-6)


def make_panel(
    base_frame: np.ndarray,
    motion_map: np.ndarray,
    scale: float,
    label: str,
) -> np.ndarray:
    label_height = 28
    heatmap = motion_heatmap(motion_map, scale)
    base = cv2.cvtColor(cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.45, heatmap, 0.55, 0.0)
    panel = np.full(
        (overlay.shape[0] + label_height, overlay.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    panel[label_height:, :] = overlay
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return panel


def motion_heatmap(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / max(scale, 1e-6), 0.0, 1.0)
    image = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def format_motion_diagnostics_summary(diagnostics: MotionDiagnostics) -> str:
    missing_regions = (
        ", ".join(diagnostics.strongest_missing_regions)
        if diagnostics.strongest_missing_regions
        else "none"
    )
    return "\n".join(
        [
            "Automated motion diagnostics (optical flow; use as hints, not ground truth):",
            (
                f"- reference_motion_mean={diagnostics.reference_motion_mean:.4g}; "
                f"target_motion_mean={diagnostics.target_motion_mean:.4g}"
            ),
            (
                f"- missing_motion_ratio={diagnostics.missing_motion_ratio:.3g}; "
                f"extra_motion_ratio={diagnostics.extra_motion_ratio:.3g}"
            ),
            (
                f"- motion_direction_error={diagnostics.motion_direction_error:.3g}; "
                f"strongest_missing_regions={missing_regions}"
            ),
            "- Inspect attached heatmaps; bright missing/extra areas mark dynamic mismatches in any region or layer.",
        ]
    )
