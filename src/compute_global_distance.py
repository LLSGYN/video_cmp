from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.video import (
    bgr_to_rgb_float,
    get_video_info,
    iter_aligned_frames,
    make_timestamps,
    min_positive,
)


SUPPORTED_METRICS = ("lpips", "psnr", "ms-ssim")


@dataclass
class DistanceResult:
    metric: str
    distance: float
    score: float | None
    compared_frames: int
    start: float
    duration: float
    fps: float
    segments: list[dict[str, float | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a global distance between a target video and a ground-truth video."
    )
    parser.add_argument("--target", required=True, help="Path to the reproduced target video.")
    parser.add_argument("--gt", required=True, help="Path to the ground-truth/reference video.")
    parser.add_argument(
        "--metrics",
        "--metric",
        default="lpips",
        choices=SUPPORTED_METRICS,
        help="Distance metric. psnr reports MSE as distance and PSNR as score.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to compare in seconds. Defaults to the shared remaining duration.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sampling FPS. Defaults to the lower FPS of the two videos.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum frames to sample. 0 means no cap.",
    )
    parser.add_argument(
        "--resize",
        choices=("target-to-gt", "gt-to-target", "smallest", "none"),
        default="target-to-gt",
        help="How to handle mismatched frame sizes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for torch/pyiqa metrics.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for pyiqa metrics: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--do_report",
        action="store_true",
        help="Report the top segments with the largest distance.",
    )
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=5.0,
        help="Segment length for --do_report.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of segments in the report.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--verbose", action="store_true", help="Print metadata with the distance.")
    return parser.parse_args()


def normalize_metric(metric: str) -> str:
    return metric.replace("_", "-").lower()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class MetricComputer:
    def __init__(self, metric: str, device: str):
        self.metric = normalize_metric(metric)
        self.device = device
        self.metric_fn = None

        if self.metric in {"lpips", "ms-ssim"}:
            import pyiqa

            metric_name = "ms_ssim" if self.metric == "ms-ssim" else self.metric
            self.metric_fn = pyiqa.create_metric(metric_name, device=device)

    def compute_batch(
        self, target_frames: list[np.ndarray], gt_frames: list[np.ndarray]
    ) -> tuple[list[float], list[float | None]]:
        if self.metric == "psnr":
            return compute_psnr_batch(target_frames, gt_frames)

        import torch

        target_tensor = frames_to_tensor(target_frames, self.device)
        gt_tensor = frames_to_tensor(gt_frames, self.device)

        with torch.no_grad():
            raw = self.metric_fn(target_tensor, gt_tensor)

        scores = raw.detach().float().cpu().reshape(-1).numpy().astype(float).tolist()
        if self.metric == "lpips":
            distances = scores
        elif self.metric == "ms-ssim":
            distances = [1.0 - score for score in scores]
        else:
            raise ValueError(f"unsupported metric: {self.metric}")
        return distances, scores


def frames_to_tensor(frames: list[np.ndarray], device: str) -> Any:
    import torch

    arrays = [bgr_to_rgb_float(frame).transpose(2, 0, 1) for frame in frames]
    return torch.from_numpy(np.stack(arrays, axis=0)).to(device)


def compute_psnr_batch(
    target_frames: list[np.ndarray], gt_frames: list[np.ndarray]
) -> tuple[list[float], list[float | None]]:
    distances: list[float] = []
    scores: list[float | None] = []

    for target_frame, gt_frame in zip(target_frames, gt_frames):
        target = bgr_to_rgb_float(target_frame)
        gt = bgr_to_rgb_float(gt_frame)
        mse = float(np.mean((target - gt) ** 2))
        distances.append(mse)
        if mse == 0.0:
            scores.append(float("inf"))
        else:
            scores.append(float(10.0 * np.log10(1.0 / mse)))

    return distances, scores


def compute_distance(args: argparse.Namespace) -> DistanceResult:
    metric = normalize_metric(args.metrics)
    target_info = get_video_info(args.target)
    gt_info = get_video_info(args.gt)
    timestamps = make_timestamps(
        target_info,
        gt_info,
        start=args.start,
        duration=args.duration,
        fps=args.fps,
        max_frames=args.max_frames or None,
    )

    device = resolve_device(args.device)
    metric_computer = MetricComputer(metric, device)

    frame_distances: list[float] = []
    frame_scores: list[float | None] = []
    frame_timestamps: list[float] = []
    target_batch: list[np.ndarray] = []
    gt_batch: list[np.ndarray] = []
    timestamp_batch: list[float] = []

    def flush_batch() -> None:
        if not target_batch:
            return
        distances, scores = metric_computer.compute_batch(target_batch, gt_batch)
        if len(distances) != len(timestamp_batch):
            raise RuntimeError(
                f"{metric} returned {len(distances)} values for "
                f"{len(timestamp_batch)} frames"
            )
        frame_distances.extend(distances)
        frame_scores.extend(scores)
        frame_timestamps.extend(timestamp_batch)
        target_batch.clear()
        gt_batch.clear()
        timestamp_batch.clear()

    for timestamp, target_frame, gt_frame in iter_aligned_frames(
        args.target, args.gt, timestamps, resize=args.resize
    ):
        target_batch.append(target_frame)
        gt_batch.append(gt_frame)
        timestamp_batch.append(timestamp)
        if len(target_batch) >= max(1, args.batch_size):
            flush_batch()
    flush_batch()

    if not frame_distances:
        raise ValueError("no frames were compared")

    finite_scores = [
        score for score in frame_scores if score is not None and np.isfinite(score)
    ]
    if finite_scores:
        score = float(np.mean(finite_scores))
    elif any(score is not None and np.isinf(score) for score in frame_scores):
        score = float("inf")
    else:
        score = None

    sample_step = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    duration = float(timestamps[-1] - timestamps[0] + sample_step)
    sample_fps = args.fps or min_positive(target_info.fps, gt_info.fps) or 1.0
    segments = (
        build_segment_report(
            frame_timestamps,
            frame_distances,
            args.segment_seconds,
            args.top_k,
        )
        if args.do_report
        else []
    )

    return DistanceResult(
        metric=metric,
        distance=float(np.mean(frame_distances)),
        score=score,
        compared_frames=len(frame_distances),
        start=float(timestamps[0]),
        duration=duration,
        fps=float(sample_fps),
        segments=segments,
    )


def build_segment_report(
    timestamps: list[float],
    distances: list[float],
    segment_seconds: float,
    top_k: int,
) -> list[dict[str, float | int]]:
    if segment_seconds <= 0:
        raise ValueError("--segment_seconds must be positive")

    buckets: dict[int, list[float]] = {}
    for timestamp, distance in zip(timestamps, distances):
        bucket = int(timestamp // segment_seconds)
        buckets.setdefault(bucket, []).append(distance)

    segments: list[dict[str, float | int]] = []
    for bucket, values in buckets.items():
        start = bucket * segment_seconds
        end = start + segment_seconds
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "distance": float(np.mean(values)),
                "frames": len(values),
            }
        )

    segments.sort(key=lambda item: float(item["distance"]), reverse=True)
    return segments[: max(0, top_k)]


def result_to_dict(result: DistanceResult) -> dict[str, Any]:
    return {
        "metric": result.metric,
        "distance": result.distance,
        "score": result.score,
        "compared_frames": result.compared_frames,
        "start": result.start,
        "duration": result.duration,
        "fps": result.fps,
        "segments": result.segments,
    }


def print_result(
    result: DistanceResult, *, as_json: bool, verbose: bool, do_report: bool
) -> None:
    if as_json:
        print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))
        return

    if not verbose and not do_report:
        print(format_float(result.distance))
        return

    print(f"metric: {result.metric}")
    print(f"distance: {format_float(result.distance)}")
    if result.score is not None:
        print(f"score: {format_float(result.score)}")
    print(f"compared_frames: {result.compared_frames}")

    if do_report:
        print("top_segments:")
        for index, segment in enumerate(result.segments, start=1):
            print(
                f"{index}. {segment['start']:.3f}s-{segment['end']:.3f}s "
                f"distance={format_float(float(segment['distance']))} "
                f"frames={segment['frames']}"
            )


def format_float(value: float) -> str:
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.8g}"


def main() -> None:
    args = parse_args()
    try:
        result = compute_distance(args)
        print_result(result, as_json=args.json, verbose=args.verbose, do_report=args.do_report)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
