"""Drop-in stand-in for the legacy ``mp.solutions.pose`` API.

MediaPipe 0.10.30+ removed ``mp.solutions``. This wraps the Tasks
``PoseLandmarker`` (VIDEO mode, "full" model = legacy model_complexity=1) so
existing code keeps working with only an import change:

    import pose_compat as mpPose
    pose = mpPose.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.85)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark): ...
"""

import os
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    # Tasks API needs a .task model file; download it once on first use
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        os.close(fd)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


class Pose:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_path=None):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, image_rgb, timestamp_ms=None):
        # VIDEO mode requires strictly increasing timestamps; default to the wall clock
        ts = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(image, ts)
        landmarks = SimpleNamespace(landmark=result.pose_landmarks[0]) if result.pose_landmarks else None
        return SimpleNamespace(pose_landmarks=landmarks)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_landmarks(image, pose_landmarks, connections=POSE_CONNECTIONS):
    # Accepts the object returned in results.pose_landmarks (or None)
    if pose_landmarks is None:
        return
    mp.tasks.vision.drawing_utils.draw_landmarks(image, pose_landmarks.landmark, connections)
