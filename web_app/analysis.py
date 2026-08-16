"""Self-contained video analysis used by Erg.ai Studio.

This intentionally mirrors the repository's core biomechanics ideas without
importing the executable scripts in main.py or toolbox.py.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

LANDMARKS = (12, 14, 16, 18, 24, 26, 28)
CONNECTIONS = ((12, 14), (14, 16), (16, 18), (12, 24), (24, 26), (26, 28))


def _ema(current: tuple[int, int], previous: tuple[int, int], alpha: float = 0.3) -> tuple[int, int]:
    return (int(current[0] * alpha + previous[0] * (1 - alpha)), int(current[1] * alpha + previous[1] * (1 - alpha)))


def _angles(points: dict[int, tuple[float, float]]) -> tuple[float, float]:
    hip, knee, ankle, shoulder = (points[24], points[26], points[28], points[12])
    a, b, c = math.dist(knee, ankle), math.dist(ankle, hip), math.dist(knee, hip)
    cosine = max(-1.0, min(1.0, (a * a - b * b + c * c) / (2 * a * c)))
    knee_angle = math.degrees(math.acos(cosine))
    body_length = math.dist(shoulder, hip)
    vertical = shoulder[1] - hip[1]
    body_angle = 180 - math.degrees(math.acos(max(-1.0, min(1.0, vertical / body_length))))
    return knee_angle, body_angle * (1 if shoulder[0] > hip[0] else -1 if shoulder[0] < hip[0] else 0)


def _draw_overlay(frame: np.ndarray, points: dict[int, tuple[int, int]], knee_angle: float, body_angle: float, stroke_count: int, stroke_rate: float | None) -> None:
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (74, 222, 128), 4, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(frame, point, 7, (101, 190, 255), cv2.FILLED, cv2.LINE_AA)

    hip, knee, ankle, shoulder = points[24], points[26], points[28], points[12]
    knee_radius = max(18, int(min(math.dist(knee, hip), math.dist(knee, ankle)) / 3))
    body_radius = max(18, int(math.dist(hip, shoulder) / 3))
    cv2.ellipse(frame, knee, (knee_radius, knee_radius), 0, 0, knee_angle, (70, 190, 255), 2, cv2.LINE_AA)
    cv2.ellipse(frame, hip, (body_radius, body_radius), -90, 0, body_angle, (70, 190, 255), 2, cv2.LINE_AA)

    panel = frame.copy()
    cv2.rectangle(panel, (24, 24), (415, 180), (10, 22, 37), cv2.FILLED)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "ERG.AI  /  LIVE ANALYSIS", (44, 57), font, 0.52, (190, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Knee angle  {knee_angle:5.1f} deg", (44, 92), font, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Body angle  {body_angle:5.1f} deg", (44, 122), font, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    rate = "--" if stroke_rate is None else f"{stroke_rate:.1f} spm"
    cv2.putText(frame, f"Stroke {stroke_count}/10  |  {rate}", (44, 153), font, 0.57, (101, 224, 172), 1, cv2.LINE_AA)


def _normalise(stroke: dict[str, list[list[float]]]) -> dict[int, dict[int, tuple[float, float]]]:
    output: dict[int, dict[int, tuple[float, float]]] = {}
    for landmark in (12, 24, 26, 28):
        values = np.asarray(stroke[str(landmark)], dtype=float)
        old = np.linspace(0, 1, len(values))
        new = np.linspace(0, 1, 100)
        output[landmark] = {index: (float(np.interp(t, old, values[:, 0])), float(np.interp(t, old, values[:, 1]))) for index, t in enumerate(new)}
    return output


def _reference_metrics(reference_path: Path, user_stroke: dict[str, list[list[float]]]) -> dict[str, float]:
    with reference_path.open(encoding="utf-8") as file:
        batches = json.load(file)
    reference = next(stroke for batch in batches if batch for stroke in batch if all(str(key) in stroke for key in (12, 24, 26, 28)))
    user, ref = _normalise(user_stroke), _normalise(reference)
    user_knee: list[float] = []
    user_body: list[float] = []
    ref_knee: list[float] = []
    ref_body: list[float] = []
    for index in range(100):
        uk, ub = _angles({key: user[key][index] for key in user})
        rk, rb = _angles({key: ref[key][index] for key in ref})
        user_knee.append(uk)
        user_body.append(ub)
        ref_knee.append(rk)
        ref_body.append(rb)
    knee_delta = np.abs(np.asarray(user_knee) - np.asarray(ref_knee))
    body_delta = np.abs(np.asarray(user_body) - np.asarray(ref_body))
    return {
        "peak_knee_deviation": round(float(knee_delta.max()), 1),
        "peak_knee_at": int(knee_delta.argmax()),
        "peak_body_deviation": round(float(body_delta.max()), 1),
        "peak_body_at": int(body_delta.argmax()),
        "knee_rmse": round(float(np.sqrt(np.mean(knee_delta**2))), 1),
        "body_rmse": round(float(np.sqrt(np.mean(body_delta**2))), 1),
        "user_finish": int(np.argmin(user_body)) + 1,
        "reference_finish": int(np.argmin(ref_body)) + 1,
    }


def coaching_feedback(metrics: dict[str, float]) -> tuple[str, bool]:
    """Return GPT coaching when configured, otherwise an informative local fallback."""
    fallback = (
        f"Your largest body-angle difference is {metrics['peak_body_deviation']}° at {metrics['peak_body_at']}% of the stroke. "
        f"Your knee-angle RMSE is {metrics['knee_rmse']}°. Focus on one change: keep the body sequence deliberate through the drive, "
        f"then compare the next ten strokes against this baseline."
    )
    if not os.getenv("OPENAI_API_KEY"):
        return fallback, False
    try:
        from openai import OpenAI

        prompt = f"""You are an experienced rowing coach. Give exactly 1-2 concise, practical cues based on this comparison to a reference stroke.
Metrics: peak knee deviation {metrics['peak_knee_deviation']}° at {metrics['peak_knee_at']}%; peak body deviation {metrics['peak_body_deviation']}° at {metrics['peak_body_at']}%; knee RMSE {metrics['knee_rmse']}°; body RMSE {metrics['body_rmse']}°; user finish {metrics['user_finish']}%, reference finish {metrics['reference_finish']}%.
Use a supportive coach voice, include timing/numbers where useful, and do not call it a data report."""
        response = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            messages=[{"role": "system", "content": "You coach rowing technique with precise, safe cues."}, {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or fallback, True
    except Exception:
        return fallback, False


def analyse_video(source: Path, destination: Path, reference_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        raise ValueError("The uploaded file could not be decoded as a video.")
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    smooth: dict[int, tuple[int, int]] = {}
    current: dict[str, list[list[float]]] = {str(key): [] for key in LANDMARKS}
    strokes: list[dict[str, list[list[float]]]] = []
    direction: int | None = None
    phase_frames = 0
    last_turn: int | None = None
    turns_since_collection_start = 0
    frame_index = 0
    stroke_rate: float | None = None
    try:
        while len(strokes) < 10:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_landmarks:
                writer.write(frame)
                frame_index += 1
                continue
            landmarks = results.pose_landmarks.landmark
            right_score = sum(landmarks[index].visibility for index in LANDMARKS)
            left_indices = (11, 13, 15, 17, 23, 25, 27)
            left_score = sum(landmarks[index].visibility for index in left_indices)
            use_right = right_score >= left_score
            ids = LANDMARKS if use_right else left_indices
            mapped = dict(zip(LANDMARKS, ids))
            if min(landmarks[index].visibility for index in ids) < 0.55:
                writer.write(frame)
                frame_index += 1
                continue
            points: dict[int, tuple[int, int]] = {}
            raw: dict[int, tuple[float, float]] = {}
            for canonical, original in mapped.items():
                # Mirror left-side input for comparison with right-side reference data.
                normalised_x = landmarks[original].x if use_right else 1 - landmarks[original].x
                raw[canonical] = (normalised_x, landmarks[original].y)
                point = (int(landmarks[original].x * width), int(landmarks[original].y * height))
                if canonical in smooth:
                    point = _ema(point, smooth[canonical])
                smooth[canonical] = point
                points[canonical] = point
                current[str(canonical)].append([raw[canonical][0], raw[canonical][1]])
            shoulder_x = points[12][0]
            previous_x = smooth.get(-1)
            smooth[-1] = (shoulder_x, 0)
            if previous_x is not None:
                movement = shoulder_x - previous_x[0]
                candidate = 1 if movement > 1 else -1 if movement < -1 else 0
                if candidate:
                    phase_frames += 1
                    if direction is None:
                        direction = candidate
                    elif candidate != direction and phase_frames >= 5:
                        if last_turn is None:
                            # Begin recording at a detected endpoint, not midway through a cycle.
                            last_turn = frame_index
                            current = {str(key): [] for key in LANDMARKS}
                            turns_since_collection_start = 0
                        else:
                            turns_since_collection_start += 1
                            if turns_since_collection_start == 2:
                                duration = frame_index - last_turn
                                stroke_rate = 60 * fps / duration if duration else None
                                strokes.append(current)
                                current = {str(key): [] for key in LANDMARKS}
                                last_turn = frame_index
                                turns_since_collection_start = 0
                        direction = candidate
                        phase_frames = 0
            knee_angle, body_angle = _angles(points)
            _draw_overlay(frame, points, knee_angle, body_angle, len(strokes), stroke_rate)
            writer.write(frame)
            frame_index += 1
    finally:
        pose.close()
        cap.release()
        writer.release()
    if len(strokes) < 10:
        raise ValueError(f"Only {len(strokes)} complete strokes were detected. Upload a clear side-view video with at least 10 strokes.")
    metrics = _reference_metrics(reference_path, strokes[-1])
    feedback, generated = coaching_feedback(metrics)
    return {"strokes": len(strokes), "metrics": metrics, "feedback": feedback, "generated_feedback": generated}
