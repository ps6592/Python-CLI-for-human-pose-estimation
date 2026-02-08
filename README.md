# Pose Estimation Pipeline


Single-person pose estimation tool that processes iPhone-style video and outputs an annotated video with skeleton overlay plus a JSON file containing 17 COCO-style body keypoints per frame.

## Quick Start

```bash
pip install -r requirements.txt
python run.py --input video.mp4 --output out.mp4 --json out.json
```

## Model Choice

**MoveNet SinglePose Lightning (TFLite, INT8)**

- Designed for single-person, real-time pose estimation
- Outputs 17 COCO keypoints natively — no mapping required
- INT8 quantized for fast CPU inference (~15-30ms per frame on modern hardware)
- Auto-downloaded from TensorFlow Hub on first run (~3 MB)

## Architecture

```
video.mp4
  │
  ▼
cv2.VideoCapture (decode)
  │
  ▼
Resize to 192×192 → MoveNet Lightning (TFLite) → 17 keypoints (y, x, score)
  │
  ▼
Scale to pixel coords → Draw skeleton + keypoints → cv2.VideoWriter
  │
  ▼
out.mp4 + out.json
```

The pipeline is single-threaded and processes one frame at a time: decode → preprocess → inference → postprocess → draw → encode.

## Output Format

### Annotated Video

- Same resolution and FPS as input
- Green dots for keypoints (score ≥ 0.2)
- Cyan skeleton lines connecting joints
- Per-frame latency/FPS overlay in top-left corner

### JSON Schema

```json
{
  "meta": { "fps": 30, "width": 1920, "height": 1080, "frame_count": 900 },
  "keypoint_format": ["nose", "left_eye", ...],
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "pose_score": 0.92,
      "keypoints": [
        { "name": "nose", "x": 960.2, "y": 210.8, "score": 0.99 },
        ...
      ]
    }
  ]
}
```

### Confidence Convention

Each keypoint `score` is in [0, 1] and represents the model's confidence that the keypoint is visible and correctly localized. This is the raw MoveNet output score. Keypoints with score < 0.2 are still reported in JSON but are not drawn on the overlay. This threshold is analogous to filtering by COCO visibility flag `v > 0`.

## Keypoints (COCO 17)

| Index | Name            |
|-------|-----------------|
| 0     | nose            |
| 1     | left_eye        |
| 2     | right_eye       |
| 3     | left_ear        |
| 4     | right_ear       |
| 5     | left_shoulder   |
| 6     | right_shoulder  |
| 7     | left_elbow      |
| 8     | right_elbow     |
| 9     | left_wrist      |
| 10    | right_wrist     |
| 11    | left_hip        |
| 12    | right_hip       |
| 13    | left_knee       |
| 14    | right_knee      |
| 15    | left_ankle      |
| 16    | right_ankle     |

## Latency

Typical per-frame latency on a modern developer machine (M-series Mac or recent Intel i7):

| Stage          | Time     |
|----------------|----------|
| Decode (cv2)   | ~2-5 ms  |
| Preprocess     | ~1-2 ms  |
| Inference      | ~15-30 ms|
| Postprocess    | ~<1 ms   |
| Draw + Encode  | ~5-10 ms |
| **Total**      | **~25-45 ms** |

Well under the 1-second requirement.

## Pushing Toward ~500 ms Total Pipeline Time (Not Per-Frame)

To push toward ~500 ms *total* for a 30-second clip (i.e., real-time or faster):

1. **Batch decoding with hardware acceleration**: Use `ffmpeg` with hardware-accelerated decoding (VideoToolbox on macOS, NVDEC on Linux) to decode frames in bulk or in a background thread, eliminating decode stalls.

2. **Pipeline parallelism**: Overlap decode, inference, and encode in a 3-stage pipeline using threads or `asyncio`. While frame N is being encoded, frame N+1 is being inferred, and frame N+2 is being decoded.

3. **GPU inference**: Run MoveNet on GPU via TensorFlow GPU, TensorRT (NVIDIA), or CoreML (Apple). GPU inference can bring per-frame time down to ~5 ms.

4. **Frame skipping with interpolation**: For low-motion segments, run inference every 2nd or 3rd frame and linearly interpolate keypoints for skipped frames. This cuts inference cost by 2-3x with minimal quality loss.

5. **Batch inference**: Accumulate N frames and run inference in a single batched call. Most accelerators are more efficient with batches.

Combined, these optimizations could realistically bring per-frame cost to ~5-10 ms, enabling real-time processing at 30 fps.

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- TensorFlow (for TFLite runtime)

## Test Video

Test videos are not included in this repo. Use any iPhone-recorded video (1080p, 25-30 fps, single person).


Pushing towards 500 ms:

One option would be pruning, but Movenet is already lean and well optimized. Deployment with threading for parallelism (on cpu) would be a great option to reduce the total processing and inference time. Determine the optimal thresholding for frame skipping to reduce latency. 
