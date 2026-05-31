from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


def get_video_info(path: str | Path) -> VideoInfo:
    video_path = Path(path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        if width <= 0 or height <= 0:
            raise ValueError(f"video has invalid dimensions: {video_path}")

        return VideoInfo(
            path=video_path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
        )
    finally:
        cap.release()


def common_duration(target: VideoInfo, gt: VideoInfo) -> float:
    if target.duration > 0 and gt.duration > 0:
        return min(target.duration, gt.duration)
    if target.fps > 0 and gt.fps > 0 and target.frame_count > 0 and gt.frame_count > 0:
        return min(target.frame_count / target.fps, gt.frame_count / gt.fps)
    return 0.0


def make_timestamps(
    target: VideoInfo,
    gt: VideoInfo,
    *,
    start: float = 0.0,
    duration: float | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
) -> np.ndarray:
    if start < 0:
        raise ValueError("--start must be non-negative")

    available_duration = common_duration(target, gt)
    if available_duration <= 0:
        raise ValueError("could not determine a common video duration")
    if start >= available_duration:
        raise ValueError(
            f"--start {start:.3f}s is beyond common duration {available_duration:.3f}s"
        )

    remaining = available_duration - start
    compare_duration = remaining if duration is None else min(duration, remaining)
    if compare_duration <= 0:
        raise ValueError("--duration must be positive")

    sample_fps = fps if fps is not None else min_positive(target.fps, gt.fps) or 1.0
    if sample_fps <= 0:
        raise ValueError("--fps must be positive")

    frame_count = max(1, int(math.floor(compare_duration * sample_fps)))
    if max_frames is not None and max_frames > 0:
        frame_count = min(frame_count, max_frames)

    step = compare_duration / frame_count
    return start + (np.arange(frame_count, dtype=np.float64) + 0.5) * step


def min_positive(*values: float) -> float | None:
    positives = [value for value in values if value > 0]
    return min(positives) if positives else None


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"could not open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def read_at(self, timestamp: float) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame

        if self.fps > 0:
            frame_index = max(0, int(round(timestamp * self.fps)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame

        raise ValueError(f"could not read frame at {timestamp:.3f}s from {self.path}")


def resize_pair(
    target_frame: np.ndarray,
    gt_frame: np.ndarray,
    *,
    mode: str = "target-to-gt",
) -> tuple[np.ndarray, np.ndarray]:
    if target_frame.shape[:2] == gt_frame.shape[:2]:
        return target_frame, gt_frame

    if mode == "none":
        raise ValueError(
            "video frame sizes differ; use --resize target-to-gt, gt-to-target, or smallest"
        )
    if mode == "target-to-gt":
        return resize_to(target_frame, gt_frame.shape[1], gt_frame.shape[0]), gt_frame
    if mode == "gt-to-target":
        return target_frame, resize_to(gt_frame, target_frame.shape[1], target_frame.shape[0])
    if mode == "smallest":
        target_area = target_frame.shape[0] * target_frame.shape[1]
        gt_area = gt_frame.shape[0] * gt_frame.shape[1]
        if target_area <= gt_area:
            return target_frame, resize_to(gt_frame, target_frame.shape[1], target_frame.shape[0])
        return resize_to(target_frame, gt_frame.shape[1], gt_frame.shape[0]), gt_frame

    raise ValueError(f"unknown resize mode: {mode}")


def resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = (
        cv2.INTER_AREA
        if frame.shape[1] > width or frame.shape[0] > height
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def iter_aligned_frames(
    target_path: str | Path,
    gt_path: str | Path,
    timestamps: Sequence[float],
    *,
    resize: str = "target-to-gt",
) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    with VideoReader(target_path) as target_reader, VideoReader(gt_path) as gt_reader:
        for timestamp in timestamps:
            target_frame = target_reader.read_at(float(timestamp))
            gt_frame = gt_reader.read_at(float(timestamp))
            target_frame, gt_frame = resize_pair(target_frame, gt_frame, mode=resize)
            yield float(timestamp), target_frame, gt_frame


def bgr_to_rgb_float(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0
