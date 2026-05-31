# video-cmp

Compare a reproduced video against a ground-truth reference video.

The project provides two workflows:

- VLM-based dynamic bug description with Google AI Studio / Gemini.
- Numeric global distance measurement with `lpips`, `psnr`, or `ms-ssim`.

## Requirements

- Python 3.11+
- `uv`
- A Google AI Studio API key for VLM comparison

Install dependencies:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root and set one of:

```bash
GEMINI_API_KEY=your_google_ai_studio_key
# or
GOOGLE_API_KEY=your_google_ai_studio_key
# or
API_KEY=your_google_ai_studio_key
```

The default Gemini model is `gemini-3.1-flash-lite`.

## VLM Clip Comparison

This command builds a short side-by-side MP4 where `target` is on the left and
`gt` is on the right, then sends it to Gemini as inline video.

```bash
uv run python src/query_clip_difference.py \
  --target /path/to/target_video.mp4 \
  --gt /path/to/ground_truth.mp4 \
  --duration 1 \
  --request_fps 15
```

Gemini receives the video part with `videoMetadata.fps`. `--request_fps` must be
at least `15` so motion issues are visible enough for animation QA.

Useful options:

- `--start`: clip start time in seconds.
- `--duration`: clip duration in seconds, default `1`.
- `--request_fps`: generated clip FPS and Gemini video metadata FPS, default `15`.
- `--model`: Gemini model name, default `gemini-3.1-flash-lite`.
- `--max_video_width`: maximum width of the generated side-by-side video.
- `--max_video_bytes`: maximum inline video size, default `20MB`.
- `--out`: write the generated comparison MP4 to a specific path.
- `--dry_run`: generate the MP4 without calling Gemini.
- `--json`: print the raw Gemini response.

Default prompt focus:

```text
Ignore static image quality. Focus only on object motion, animation continuity,
and timeline alignment. The target reproduction is on the left, and the official
ground-truth reference is on the right. Identify the single most severe dynamic
issue in the target video using production terminology.
```

## Global Distance

Compute a single numeric distance between the two videos:

```bash
uv run python src/compute_global_distance.py \
  --target /path/to/target_video.mp4 \
  --gt /path/to/ground_truth.mp4 \
  --metrics lpips
```

Supported metrics:

- `lpips`: perceptual distance, default.
- `psnr`: reports MSE as `distance` and PSNR as `score`.
- `ms-ssim`: reports `1 - ms_ssim` as `distance`.

Useful options:

- `--start`: comparison start time in seconds.
- `--duration`: comparison duration in seconds.
- `--fps`: sampling FPS. Defaults to the lower FPS of the two videos.
- `--max_frames`: cap sampled frames for faster checks.
- `--resize`: frame size alignment mode, default `target-to-gt`.
- `--device`: `auto`, `cpu`, `cuda`, etc. for `pyiqa` metrics.
- `--json`: emit structured JSON.
- `--verbose`: print metadata with the result.

## Segment Report

Use `--do_report` to find the most different time ranges:

```bash
uv run python src/compute_global_distance.py \
  --target /path/to/target_video.mp4 \
  --gt /path/to/ground_truth.mp4 \
  --metrics psnr \
  --do_report \
  --segment_seconds 5 \
  --top_k 3
```

## Notes

- All comparisons are aligned by timestamp over the shared duration of the two videos.
- If video sizes differ, frames are resized according to `--resize`.
- LPIPS may download model weights on first use through `pyiqa` / PyTorch.
- Inline Gemini video requests are size-checked before sending. Use a shorter clip
  or lower `--max_video_width` if the generated MP4 exceeds the limit.
