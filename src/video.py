"""Frame-by-frame video processing used by the Streamlit app."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from src.detector import AeroVisionDetector


def iter_annotated_frames(
    detector: AeroVisionDetector, video_path: str | Path
) -> Iterator[tuple[np.ndarray, Counter]]:
    """Read a video file frame by frame, yielding the annotated frame and its counts."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            annotated_frame, counts = detector.predict_frame(frame)
            yield annotated_frame, counts
    finally:
        capture.release()


def get_video_info(video_path: str | Path) -> dict:
    """Return basic metadata (fps, frame count, size) for a video file."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info
