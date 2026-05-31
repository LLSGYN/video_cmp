from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

from utils.video import (
    get_video_info,
    iter_aligned_frames,
    make_timestamps,
    resize_to,
)


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_REQUEST_FPS = 15.0
DEFAULT_MAX_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
DEFAULT_PROMPT = """Act as a senior animator and game QA specialist. Carefully compare the two side-by-side video clips.

Ignore static image quality. Focus only on object motion, animation continuity, and timeline alignment.
The target reproduction is on the left, and the official ground-truth reference is on the right.
The reference motion is correct. Identify the single most severe dynamic issue in the target video, such as delayed motion, high-frequency local jitter, incorrect animation state transitions, timing drift, or temporal discontinuity.
Describe the dynamic bug using clear production terminology."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a VLM to describe the most important dynamic difference in a video clip."
    )
    parser.add_argument("--target", required=True, help="Path to the reproduced target video.")
    parser.add_argument("--gt", required=True, help="Path to the ground-truth/reference video.")
    parser.add_argument("--start", type=float, default=0.0, help="Clip start time in seconds.")
    parser.add_argument("--duration", type=float, default=1.0, help="Clip duration in seconds.")
    parser.add_argument(
        "--request_fps",
        type=float,
        default=DEFAULT_REQUEST_FPS,
        help="FPS used for the generated comparison clip and Gemini videoMetadata.",
    )
    parser.add_argument(
        "--resize",
        choices=("target-to-gt", "gt-to-target", "smallest", "none"),
        default="target-to-gt",
        help="How to handle mismatched frame sizes before placing videos side by side.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Google AI Studio Gemini model name.",
    )
    parser.add_argument("--api_key", default=None, help="API key. Defaults to env or .env.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the VLM.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum response tokens.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--max_video_width",
        type=int,
        default=1600,
        help="Maximum width of the generated side-by-side video.",
    )
    parser.add_argument(
        "--max_video_bytes",
        type=int,
        default=DEFAULT_MAX_INLINE_VIDEO_BYTES,
        help="Maximum inline video request size in bytes.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path for the generated comparison video.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only generate the comparison video; do not call the VLM.",
    )
    parser.add_argument("--json", action="store_true", help="Print the raw response JSON.")
    return parser.parse_args()


def build_comparison_video(args: argparse.Namespace) -> Path:
    assert_request_fps(args.request_fps)

    target_info = get_video_info(args.target)
    gt_info = get_video_info(args.gt)
    timestamps = make_timestamps(
        target_info,
        gt_info,
        start=args.start,
        duration=args.duration,
        fps=args.request_fps,
    )

    output_path = comparison_video_path(args.out)
    writer: cv2.VideoWriter | None = None
    written_frames = 0

    try:
        for timestamp, target_frame, gt_frame in iter_aligned_frames(
            args.target, args.gt, timestamps, resize=args.resize
        ):
            frame = make_labeled_frame(
                target_frame,
                gt_frame,
                timestamp,
                args.max_video_width,
            )
            if writer is None:
                writer = open_video_writer(
                    output_path,
                    frame.shape[1],
                    frame.shape[0],
                    args.request_fps,
                )
            writer.write(frame)
            written_frames += 1
    finally:
        if writer is not None:
            writer.release()

    if written_frames == 0:
        raise ValueError("no frames were sampled")

    assert_video_source_size(output_path, args.max_video_bytes)
    return output_path


def assert_request_fps(request_fps: float) -> None:
    if request_fps < DEFAULT_REQUEST_FPS:
        raise ValueError(f"--request_fps must be at least {DEFAULT_REQUEST_FPS:g}")


def comparison_video_path(out_path: str | None) -> Path:
    if out_path:
        path = Path(out_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    handle = tempfile.NamedTemporaryFile(prefix="video_cmp_", suffix=".mp4", delete=False)
    handle.close()
    return Path(handle.name)


def open_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create comparison video: {path}")
    return writer


def make_labeled_frame(
    target_frame: np.ndarray,
    gt_frame: np.ndarray,
    timestamp: float,
    max_video_width: int,
) -> np.ndarray:
    label_height = 34
    gap = 8
    max_pair_width = max(320, max_video_width)

    pair_width = target_frame.shape[1] + gt_frame.shape[1] + gap
    if pair_width > max_pair_width:
        scale = (max_pair_width - gap) / max(
            1, target_frame.shape[1] + gt_frame.shape[1]
        )
        target_frame = resize_to(
            target_frame,
            max(1, int(target_frame.shape[1] * scale)),
            max(1, int(target_frame.shape[0] * scale)),
        )
        gt_frame = resize_to(
            gt_frame,
            max(1, int(gt_frame.shape[1] * scale)),
            max(1, int(gt_frame.shape[0] * scale)),
        )

    row_width = target_frame.shape[1] + gt_frame.shape[1] + gap
    row_height = max(target_frame.shape[0], gt_frame.shape[0]) + label_height
    row_width = make_even(row_width)
    row_height = make_even(row_height)
    frame = np.full((row_height, row_width, 3), 255, dtype=np.uint8)
    target_x = 0
    gt_x = target_frame.shape[1] + gap
    cv2.putText(
        frame,
        f"target left | t={timestamp:.3f}s",
        (target_x + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "ground truth right",
        (gt_x + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    frame[
        label_height : label_height + target_frame.shape[0],
        target_x : target_x + target_frame.shape[1],
    ] = target_frame
    frame[
        label_height : label_height + gt_frame.shape[0],
        gt_x : gt_x + gt_frame.shape[1],
    ] = gt_frame
    return frame


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def assert_video_source_size(path: Path, max_bytes: int) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"inline video source is {size} bytes, exceeding --max_video_bytes "
            f"{max_bytes}; use a shorter clip or lower --max_video_width"
        )


def call_google_ai_studio(args: argparse.Namespace, video_path: Path) -> dict:
    api_key = args.api_key or find_api_key()
    if not api_key:
        raise RuntimeError(
            "missing API key; set GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY "
            "in the environment or .env"
    )

    model = args.model
    model_name = urllib.parse.quote(model.removeprefix("models/"), safe="")
    endpoint = DEFAULT_ENDPOINT.format(model=model_name)
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "video/mp4",
                            "data": video_b64,
                        },
                        "videoMetadata": {
                            "fps": args.request_fps,
                        },
                    },
                    {"text": args.prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_tokens,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Google AI Studio HTTP {error.code}: {summarize_google_error(body)}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Google AI Studio request failed: {error}") from error


def find_api_key() -> str | None:
    key_names = (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_AI_STUDIO_API_KEY",
        "API_KEY",
    )
    for key in key_names:
        value = os.environ.get(key)
        if value:
            return value

    env_paths = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for env_path in dict.fromkeys(path for path in env_paths if path.exists()):
        values = read_dotenv(env_path)
        for key in key_names:
            value = values.get(key)
            if value:
                return value
    return None


def summarize_google_error(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body

    error = data.get("error")
    if not isinstance(error, dict):
        return body

    message = str(error.get("message") or "request failed")
    status = error.get("status")
    if status:
        return f"{message}; status={status}"
    return message


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def extract_answer(response: dict) -> str:
    try:
        candidates = response["candidates"]
        if not candidates:
            raise RuntimeError(f"empty Gemini candidates: {response}")
        content = candidates[0]["content"]
        parts = content.get("parts", [])
    except (KeyError, TypeError) as exc:
        prompt_feedback = response.get("promptFeedback")
        if prompt_feedback:
            raise RuntimeError(f"Gemini prompt feedback: {prompt_feedback}") from exc
        raise RuntimeError(f"unexpected Gemini response shape: {response}") from exc

    texts = [
        str(part["text"]) for part in parts if isinstance(part, dict) and part.get("text")
    ]
    if texts:
        return "\n".join(texts).strip()

    finish_reason = candidates[0].get("finishReason")
    raise RuntimeError(f"Gemini returned no text; finish_reason={finish_reason}")


def main() -> None:
    args = parse_args()
    try:
        video_path = build_comparison_video(args)

        if args.dry_run:
            print(str(video_path))
            return

        response = call_google_ai_studio(args, video_path)
        if args.json:
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return

        print(extract_answer(response).strip())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
