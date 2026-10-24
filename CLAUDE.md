# CLAUDE.md

Erg.ai is a computer-vision rowing (erg) technique analyzer. It runs MediaPipe Pose over side-view erg video, smooths the landmarks with an exponential moving average (EMA), detects strokes from shoulder movement, computes knee and body angles, and compares a user stroke against reference strokes taken from YouTube videos. It can also generate coaching cues through the OpenAI API.

The repo has two separate halves:

1. **Root scripts**: research and prototype code. These are standalone scripts with no package structure, a lot of module-level global state and hardcoded paths, and work-in-progress TODOs.
2. **`web_app/`**: a self-contained FastAPI app ("Erg.ai Studio"). It **re-implements** the core logic and deliberately does not import `main.py` or `toolbox.py`. If you change the math in one half, check whether the other half needs the same change.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Live OpenCV-window prototype. Runs on `test_dir/satvik_erg.mp4`, draws the skeleton, angle arcs, angular velocity, stroke rate and count. Appends each detected stroke's pixel coordinates to `user_strokes.json`. Logic sits directly under `if __name__ == '__main__'` (marked `#TODO: Refactor!!`). |
| `pose_compat.py` | Shim with the legacy `mp.solutions.pose` interface (`Pose(...).process(rgb)` → `.pose_landmarks.landmark`, `draw_landmarks`, `POSE_CONNECTIONS`), built on the MediaPipe Tasks `PoseLandmarker` in VIDEO mode. Auto-downloads `models/pose_landmarker_full.task` (gitignored). All root scripts use it as `mpPose`/`mpDraw`. |
| `toolbox.py` | Analysis helpers: angle math (`calc_angles`), time-normalization to 100 samples (`normalize_t`, `process_poses`, `stack_poses`), `compare_ref` (per-% angle deviation and RMSE), and `gpt_wrapper` (derives body open/close timing and velocities, then calls OpenAI). The real-time helpers are `StrokeDetector` (a streaming port of `main.py`'s detector), `average_angle_curves`, `load_reference_curves`, `compare_angle_curves`, and `coaching_feedback` (`gpt_wrapper` with a `local_feedback` fallback). Its `__main__` compares `user_strokes[3]` against `ten_strokes[0][3]`. |
| `stroke_counter.py` | Builds the reference dataset. `check_validity(path)` finds 10 contiguous "good" strokes (stable rate within 7 spm, shoulder travel > 0.25 of frame width) and **appends** them to `ten_strokes.json`. Runs at import time on a hardcoded `test_dir/...` video. |
| `youtube_extraction.py` | Downloads YouTube videos into `test_dir/` with `pytubefix`. The YouTube Data API search pipeline below the download call is commented out (it would need `YOUTUBE_API_KEY`). |
| `check_validity.py` | Checks whether the first video in `Videos/` shows a clear side profile by sampling 10 mid-video frames (uses `moviepy`). |
| `json_analysis.py` | matplotlib animation of a normalized reference stroke from `ten_strokes.json`. |
| `real_time.py` | Webcam (`VideoCapture(0)`) pose-overlay demo. |
| `Bluetooth_test.py` | **Live Coach** (the name is misleading). It reads the webcam, detects strokes, averages angle curves over a sliding window of the last 3 strokes, and gets a cue from OpenAI that is spoken with Google Cloud TTS using the Chirp3-HD Kore voice (authenticated with Application Default Credentials, no API key). With `sounddevice` installed, audio is streamed and plays as it arrives. Without it, the script falls back to downloading a full WAV and playing it with `afplay`/`aplay`/`winsound`. LLM text is passed through `toolbox.strip_asterisks` before it is shown or spoken. Feedback and speech run on a background `Coach` thread, one cue at a time. It is separate from `web_app/`. The PRD is in `.taskmaster/docs/prd.txt`. |
| `web_app/app.py` | FastAPI server. `POST /api/analyse` takes an `.mp4`/`.mov` upload and runs analysis in a background thread. Jobs are polled at `GET /api/jobs/{id}` and the annotated MP4 is served from `GET /api/jobs/{id}/video`. Jobs live in an in-memory dict, so they are lost on restart. |
| `web_app/analysis.py` | `analyse_video()`: pose tracking, overlay rendering, and turn-based stroke detection. It stops after 10 strokes, then compares the **last** stroke to the first complete reference in `ten_strokes.json`. `coaching_feedback()` falls back to a local, template-based message when `OPENAI_API_KEY` is unset or the API call fails. |
| `web_app/static/` | Vanilla HTML/CSS/JS frontend: upload with drag-and-drop, then poll, then show the video, metrics and feedback. |

## Running

There is no test suite, linter, or build step.

```bash
# Web app (from web_app/, so `from analysis import ...` resolves)
cd web_app
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Prototype scripts (run from repo root; they use relative paths)
python main.py          # needs test_dir/satvik_erg.mp4, opens a cv2 window
python toolbox.py       # stroke comparison + OpenAI call

# Live coach
python Bluetooth_test.py                      # webcam, spoken cues
python Bluetooth_test.py --video test_dir/satvik_erg.mp4 --headless --no-voice --offline   # deterministic smoke test
python Bluetooth_test.py --say "test"         # TTS only
```

- Secrets live in a root `.env` (gitignored) and are loaded by `python-dotenv` in `toolbox.py`. The only key is `OPENAI_API_KEY`. Google TTS uses ADC instead: run `gcloud auth application-default login` and `gcloud auth application-default set-quota-project <project-id>` once. The quota project is required, because user credentials without one get a 403. `web_app` reads `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default `gpt-5.4-mini`) from the process environment. It does not load `.env`.
- MediaPipe is **1.1.x** (`mediapipe>=1.1.1,<2.0`), which removed `mp.solutions`. Root code goes through `pose_compat.py`. `web_app/analysis.py` calls the Tasks API inline to stay self-contained, using its own model download to `web_app/models/`. There is no Python version pin. A VIDEO-mode landmarker needs strictly increasing timestamps; `pose_compat` uses the wall clock unless `timestamp_ms` is passed.
- `test_dir/`, `Videos/`, `web_app/data/` and `.venv/` are gitignored. Video inputs are local only.

## Domain conventions

- **MediaPipe landmark IDs.** Right side: `12` shoulder, `14` elbow, `16` wrist, `18` pinky, `24` hip, `26` knee, `28` ankle. Left side: the same joints with odd IDs (`11, 13, 15, 17, 23, 25, 27`). Some code also tracks `8`/`7` (ears). Angle math always uses `24/26/28/12` (hip, knee, ankle, shoulder).
- **Side selection.** Whichever side has higher landmark visibility is used. `web_app` mirrors left-side x coordinates (`1 - x`) and maps them onto the right-side IDs so they match the right-facing reference data.
- **Knee angle**: law of cosines on hip, knee and ankle. **Body angle**: `180 - acos(dy/len)` of the hip→shoulder vector, signed by whether the shoulder is ahead of or behind the hip. Both `toolbox` and `web_app` clamp the `acos` inputs to [-1, 1]. `main.py`'s own copy does not.
- **EMA smoothing**: alpha of about 0.3 (a comment says this minimized angle variance; `main.py` uses 0.275).
- **Stroke normalization**: each stroke is resampled to 100 points, so an index equals "% of stroke" (0% = catch).
- **Stroke detection** has two different implementations:
  - `main.py` and `stroke_counter.py` use a finish/catch state machine. It compares the shoulder x position with its value 7 frames earlier and waits 10 frames after the finish before looking for the catch.
  - `web_app/analysis.py` counts direction reversals of the shoulder x position (at least 5 frames per phase). Two turns make one stroke, and recording starts only at the first detected turn.
  - The latest commit notes that direction switching in video analysis still needs debugging.

## Data files (large JSON; don't read them whole)

- `ten_strokes.json` (~700 KB): reference strokes, shaped `list[batch]`, where `batch` is `list[stroke]` (10 strokes per non-empty batch) and `stroke` is `{"<landmark id>": [[x, y], ...]}`. Coordinates are **normalized 0–1**, keys are strings, and some batches may be empty. Each batch comes from one YouTube video, appended by `stroke_counter.py`.
- `user_strokes.json` (~5 MB): `list[stroke]` with the same per-stroke shape but **pixel** coordinates (frames resized to 1700×1000), appended by `main.py`. Pixel and normalized data are mixed in comparisons; `toolbox.absolute_to_relative` exists for this but is marked `TODO: DEBUG`.
- Both files are **appended to** each time a script runs. Avoid running those scripts casually, or back the files up first.

## Gotchas

- Most root scripts run work at import time (`stroke_counter.py`, `check_validity.py`, `json_analysis.py`, `youtube_extraction.py`). Don't import them; copy the function you need or refactor it first.
- `main.py` duplicates the angle helpers that also live in `toolbox.py`.
- `import main` is safe because its loop sits under `__main__`. `Bluetooth_test.py` imports it as `prototype` to reuse `determine_right`, `exponential_moving_average` and `draw_pose_graph`.
- `ten_strokes.json` is normalized 0–1. `load_reference_curves` scales x by an assumed 16:9 aspect ratio before measuring angles.
- `__pycache__/` and `.idea/` files are committed even though `.gitignore` lists them.
