#!/usr/bin/env python3
"""
Pose Estimation Pipeline
========================
Takes an iPhone-style video and outputs:
  1) An annotated video with skeleton overlay
  2) A JSON file with 17 COCO-style body keypoints per frame

Usage:
    python run.py --input video.mp4 --output out.mp4 --json out.json

Model: MoveNet SinglePose Lightning (TFLite) — optimized for single-person,
       low-latency inference on CPU.

Confidence convention:
    Each keypoint has a `score` in [0, 1] representing the model's confidence
    that the keypoint is visible and correctly localized. This corresponds to
    the MoveNet output score and is analogous to COCO visibility flag v=2
    (visible and labeled). Keypoints with score < 0.2 are still reported but
    should be considered unreliable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton edges for drawing — pairs of keypoint indices
SKELETON_EDGES = [
    (0, 1), (0, 2),          # nose → eyes
    (1, 3), (2, 4),          # eyes → ears
    (5, 6),                   # shoulder → shoulder
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 11), (6, 12),        # torso sides
    (11, 12),                 # hip → hip
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]

# Colors (BGR)
KEYPOINT_COLOR = (0, 255, 0)
SKELETON_COLOR = (255, 255, 0)
TEXT_COLOR = (0, 0, 255)
KEYPOINT_RADIUS = 5
SKELETON_THICKNESS = 2
CONFIDENCE_THRESHOLD = 0.2  # minimum score to draw a keypoint/edge

# MoveNet input size (Lightning)
MOVENET_INPUT_SIZE = 192

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_movenet_model() -> tf.lite.Interpreter:
    """Download (if needed) and load MoveNet SinglePose Lightning via TF Hub."""
    model_url = (
        "https://tfhub.dev/google/lite-model/"
        "movenet/singlepose/lightning/tflite/int8/4"
        "?lite-format=tflite"
    )
    model_path = tf.keras.utils.get_file(
        "movenet_lightning_int8.tflite",
        model_url,
    )
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_movenet(
    interpreter: tf.lite.Interpreter,
    frame: np.ndarray,
) -> np.ndarray:
    """
    Run MoveNet on a single BGR frame.

    Returns:
        keypoints – np.ndarray of shape (17, 3) where columns are (y, x, score)
                    in *normalized* coordinates [0, 1].
    """
    # Prepare input: resize, convert BGR→RGB, add batch dim
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))
    input_tensor = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    # Output shape: (1, 1, 17, 3) — [batch, person, keypoint, (y, x, score)]
    keypoints = interpreter.get_tensor(output_details[0]["index"])
    return keypoints[0, 0, :, :]  # (17, 3)


def keypoints_to_pixel(
    keypoints: np.ndarray,
    width: int,
    height: int,
) -> list[dict]:
    """Convert normalized (y, x, score) keypoints to pixel-coord dicts."""
    result = []
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        y_norm, x_norm, score = keypoints[i]
        result.append({
            "name": name,
            "x": round(float(x_norm * width), 1),
            "y": round(float(y_norm * height), 1),
            "score": round(float(score), 4),
        })
    return result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton(
    frame: np.ndarray,
    keypoints_px: list[dict],
) -> np.ndarray:
    """Draw keypoints and skeleton edges on a frame (in-place)."""
    coords = [(kp["x"], kp["y"], kp["score"]) for kp in keypoints_px]

    # Draw edges first (so dots are on top)
    for i, j in SKELETON_EDGES:
        if coords[i][2] >= CONFIDENCE_THRESHOLD and coords[j][2] >= CONFIDENCE_THRESHOLD:
            pt1 = (int(coords[i][0]), int(coords[i][1]))
            pt2 = (int(coords[j][0]), int(coords[j][1]))
            cv2.line(frame, pt1, pt2, SKELETON_COLOR, SKELETON_THICKNESS)

    # Draw keypoints
    for x, y, score in coords:
        if score >= CONFIDENCE_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

    return frame


def draw_latency(frame: np.ndarray, latency_ms: float, fps: float) -> np.ndarray:
    """Overlay per-frame latency and effective FPS."""
    text = f"Latency: {latency_ms:.0f}ms | FPS: {fps:.1f}"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA,
    )
    return frame


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(
    input_path: str,
    output_path: str,
    json_path: str,
) -> None:
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{input_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {width}x{height} @ {fps:.2f} fps, {frame_count} frames")

    # Load model
    print("Loading MoveNet Lightning…")
    interpreter = load_movenet_model()

    # Warm up: run a dummy inference so the first real frame isn't penalized
    warmup_input = np.zeros((1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3), dtype=np.uint8)
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], warmup_input)
    interpreter.invoke()
    print("Model loaded and warmed up.")

    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # JSON structure
    output_data: dict = {
        "meta": {
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "frame_count": frame_count,
        },
        "keypoint_format": COCO_KEYPOINT_NAMES,
        "frames": [],
    }

    frame_idx = 0
    total_latency = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        # --- Inference ---
        keypoints_norm = run_movenet(interpreter, frame)
        keypoints_px = keypoints_to_pixel(keypoints_norm, width, height)

        # Compute per-frame pose score (mean of all keypoint scores)
        pose_score = float(np.mean([kp["score"] for kp in keypoints_px]))

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        total_latency += latency_ms
        effective_fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        # --- Draw ---
        draw_skeleton(frame, keypoints_px)
        draw_latency(frame, latency_ms, effective_fps)

        writer.write(frame)

        # --- JSON record ---
        timestamp_ms = round((frame_idx / fps) * 1000, 2) if fps > 0 else 0
        output_data["frames"].append({
            "frame_index": frame_idx,
            "timestamp_ms": timestamp_ms,
            "pose_score": round(pose_score, 4),
            "keypoints": keypoints_px,
        })

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == frame_count:
            avg = total_latency / frame_idx
            print(
                f"  Processed {frame_idx}/{frame_count} frames "
                f"(avg {avg:.1f} ms/frame)"
            )

    cap.release()
    writer.release()

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    avg_latency = total_latency / max(frame_idx, 1)
    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Average latency: {avg_latency:.1f} ms/frame")
    print(f"Output video: {output_path}")
    print(f"Output JSON:  {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose estimation pipeline — COCO 17 keypoints",
    )
    parser.add_argument(
        "--input", required=True, help="Path to input video (e.g. video.mp4)",
    )
    parser.add_argument(
        "--output", required=True, help="Path for annotated output video",
    )
    parser.add_argument(
        "--json", required=True, help="Path for JSON keypoint output",
    )
    args = parser.parse_args()

    # Validate input exists
    if not Path(args.input).is_file():
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    process_video(args.input, args.output, args.json)


if __name__ == "__main__":
    main()
