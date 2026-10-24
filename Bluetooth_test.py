"""Erg.ai Live Coach: real-time, voice-coached erg technique feedback.

Reads the camera with OpenCV, tracks strokes with main.py's detector logic
(via toolbox.StrokeDetector), averages knee/body angle curves over a sliding
window of the last few strokes, gets coaching cues from OpenAI
(toolbox.gpt_wrapper) and speaks them with Google Cloud TTS.

    python Bluetooth_test.py                      # webcam 0, spoken coaching
    python Bluetooth_test.py --video test_dir/satvik_erg.mp4 --headless --no-voice --offline
    python Bluetooth_test.py --say "Quick TTS test"
"""

import argparse
import math
import os
import platform
import shutil
import subprocess
import tempfile
import textwrap
import threading
import time
from collections import deque

import cv2
from google.cloud import texttospeech

import main as prototype  # pose helpers from the original prototype
import pose_compat
import toolbox  # also loads .env (OPENAI_API_KEY)



VOICE = "en-US-Chirp3-HD-Kore"  # ~0.6s per cue vs ~3.2s for Gemini-TTS Kore, but no style prompt
LANGUAGE = "en-US"

REFERENCE_PATH = "ten_strokes.json"
FRAME_WIDTH = 1280        # frames are resized to this width (keeps overlay text sizes sane)
EMA_ALPHA = 0.275         # main.py's smoothing value
VERT_SCALE = 1.2          # main.py's ghost body line length
LOST_POSE_RESET_S = 1.5   # rower out of frame this long -> re-pick side, drop partial stroke

_tts_client = None


# ------------------Voice------------------

def synthesize(text: str, out_path: str = "output.wav") -> str:
    # LINEAR16 is a WAV, which afplay/aplay/winsound can all play without extra deps
    # Auth is Application Default Credentials (gcloud auth application-default login),
    # billed to the ADC quota project; no API key
    global _tts_client
    if _tts_client is None:
        _tts_client = texttospeech.TextToSpeechClient()

    response = _tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code=LANGUAGE,
            name=VOICE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        ),
    )

    with open(out_path, "wb") as f:
        f.write(response.audio_content)
    return out_path


def play_audio(path: str) -> bool:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["afplay", path], check=False)
    elif system == "Linux" and shutil.which("aplay"):
        subprocess.run(["aplay", "-q", path], check=False)
    elif system == "Windows":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        return False
    return True


def speak(text: str) -> bool:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        synthesize(text, path)
        return play_audio(path)
    finally:
        os.remove(path)


# ------------------Coach------------------

class Coach:
    """Runs analysis -> feedback -> speech on a background thread, one cue at a time."""

    def __init__(self, reference, use_llm=True, voice=True):
        self.reference = reference  # (knee curve, body curve)
        self.use_llm = use_llm
        self.voice = voice
        self.status = "idle"
        self.last_feedback = None
        self.cues_given = 0
        self._thread = None

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def submit(self, strokes):
        # Ignored while a cue is still being generated or spoken, so cues never overlap
        if self.busy:
            return False
        poses = [s.poses for s in strokes]
        self._thread = threading.Thread(target=self._run, args=(poses,), daemon=True)
        self._thread.start()
        return True

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, poses):
        try:
            self.status = "analyzing"
            curves = toolbox.average_angle_curves(poses)
            if curves is None:
                return
            comparison = toolbox.compare_angle_curves(*curves, *self.reference)
            self.status = "thinking"
            text, generated = toolbox.coaching_feedback(comparison, self.use_llm)
            self.last_feedback = text
            self.cues_given += 1
            print(f"\n[coach{' (AI)' if generated else ''}] {text}\n")
            if self.voice:
                self.status = "speaking"
                try:
                    speak(text)
                except Exception as exc:
                    # e.g. missing key or no network: keep coaching on screen only
                    print(f"[coach] Speech failed, continuing without voice: {exc}")
                    self.voice = False
        finally:
            self.status = "idle"


# ------------------Pose------------------

def extract_points(landmarks, side, width, height, smooth, alpha=EMA_ALPHA):
    """Returns (display, canonical) points keyed by right-side landmark ids.

    display: EMA-smoothed pixel points for drawing.
    canonical: the same points mirrored to face right when the left side is
    visible, so they match the right-facing reference strokes.
    """
    ids = prototype.right_pose_list if side == 1 else prototype.left_pose_list
    display, canonical = {}, {}
    for canonical_id, lm_id in zip(prototype.right_pose_list, ids):
        lm = landmarks[lm_id]
        cx, cy = int(lm.x * width), int(lm.y * height)
        if canonical_id in smooth:
            px, py = smooth[canonical_id]
            cx, cy = prototype.exponential_moving_average(cx, cy, px, py, alpha)
        smooth[canonical_id] = (cx, cy)
        display[canonical_id] = (cx, cy)
        canonical[canonical_id] = (cx if side == 1 else width - cx, cy)
    return display, canonical


# ------------------Overlay------------------

def _put_text(img, text, org, scale=0.8, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_overlay(img, display, knee_angle, body_angle, detector, window_len, window_size, coach, side):
    h, w = img.shape[:2]
    if display:
        for i, f in prototype.right_connections:
            cv2.line(img, display[i], display[f], (0, 255, 0), 5)
        for point in display.values():
            cv2.circle(img, point, 10, (255, 0, 0), cv2.FILLED)
        if knee_angle is not None:
            hip, knee, ankle, shoulder = display[24], display[26], display[28], display[12]
            ghost_body_pos = (hip[0], max(0, round(hip[1] - VERT_SCALE * math.dist(hip, shoulder))))
            try:
                prototype.draw_pose_graph(img, hip, knee, ankle, shoulder, ghost_body_pos, knee_angle, body_angle)
            except (ValueError, ZeroDivisionError):
                pass  # degenerate geometry on this frame
    elif side is None:
        _put_text(img, "Please move back so your body is visible", (40, h // 2), 1.0, (0, 0, 255))

    rate = "--" if detector.last_rate is None else f"{detector.last_rate:.1f} spm"
    _put_text(img, f"Strokes: {detector.stroke_count}   Rate: {rate}", (30, 40))
    _put_text(img, f"Window: {window_len}/{window_size}   Coach: {coach.status}", (30, 75))
    if coach.last_feedback:
        lines = textwrap.wrap(coach.last_feedback, 80)[:4]
        top = h - 20 - 32 * len(lines)
        panel = img.copy()
        cv2.rectangle(panel, (0, top - 30), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(panel, 0.6, img, 0.4, 0, img)
        for i, line in enumerate(lines):
            _put_text(img, line, (30, top + 32 * i), 0.75, (180, 255, 180))


# ------------------Main loop------------------

def run(source=0, window=3, headless=False, voice=True, use_llm=True, reference_path=REFERENCE_PATH):
    reference = toolbox.load_reference_curves(reference_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {source!r}. On macOS, allow camera access for your terminal/VS Code.")
    is_file = isinstance(source, str)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = pose_compat.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.85)
    detector = toolbox.StrokeDetector()
    strokes = deque(maxlen=window)  # sliding window of the most recent valid strokes
    coach = Coach(reference, use_llm=use_llm, voice=voice)
    smooth = {}
    side = None
    last_seen = None
    frame_idx = 0
    print("Erg.ai Live Coach running. Press q in the video window to quit.")
    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            # Video files use their own clock so results don't depend on processing speed
            timestamp = frame_idx / fps if is_file else time.monotonic()
            frame_idx += 1
            h, w = img.shape[:2]
            img = cv2.resize(img, (FRAME_WIDTH, round(h * FRAME_WIDTH / w)))
            h, w = img.shape[:2]
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), timestamp_ms=timestamp * 1000)

            display = None
            knee_angle = body_angle = None
            if results.pose_landmarks:
                if side is None:
                    visible_side = prototype.determine_right(results)
                    if visible_side != -1:
                        side = visible_side
                        smooth.clear()
                        detector.reset()
                if side is not None:
                    last_seen = timestamp
                    display, canonical = extract_points(results.pose_landmarks.landmark, side, w, h, smooth)
                    try:
                        knee_angle, body_angle = toolbox.calc_angles(display[24], display[26], display[28], display[12])
                    except ZeroDivisionError:
                        pass
                    stroke = detector.update(canonical, timestamp, w)
                    if stroke:
                        strokes.append(stroke)
                        print(f"Stroke {detector.stroke_count}: {stroke.rate:.1f} spm")
                        if len(strokes) == window:
                            coach.submit(list(strokes))
            if side is not None and timestamp - last_seen > LOST_POSE_RESET_S:
                side = None
                detector.reset()

            if not headless:
                draw_overlay(img, display, knee_angle, body_angle, detector, len(strokes), window, coach, side)
                cv2.imshow("Erg.ai Live Coach", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pose.close()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    coach.join()  # let the last cue finish
    return {"strokes": detector.stroke_count, "cues": coach.cues_given}


def main():
    parser = argparse.ArgumentParser(description="Erg.ai real-time voice coach")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--video", help="analyze a video file instead of the camera")
    parser.add_argument("--window", type=int, default=3, help="strokes averaged per cue (default 3)")
    parser.add_argument("--reference", default=REFERENCE_PATH, help="reference strokes JSON")
    parser.add_argument("--headless", action="store_true", help="don't open a video window")
    parser.add_argument("--no-voice", action="store_true", help="print cues instead of speaking them")
    parser.add_argument("--offline", action="store_true", help="use local feedback instead of OpenAI")
    parser.add_argument("--say", help="just speak this text (TTS test) and exit")
    args = parser.parse_args()

    if args.say:
        speak(args.say)
        return
    summary = run(
        source=args.video if args.video else args.camera,
        window=args.window,
        headless=args.headless,
        voice=not args.no_voice,
        use_llm=not args.offline,
        reference_path=args.reference,
    )
    print(f"Done: {summary['strokes']} strokes, {summary['cues']} cues.")


if __name__ == "__main__":
    main()
