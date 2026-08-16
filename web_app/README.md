# Erg.ai Studio

An isolated web version of the rowing-video analyzer. It does not import or run
the interactive loops in the repository's `main.py` or `toolbox.py`.

## Run locally

```powershell
cd web_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "..." # optional; enables generated coaching feedback
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Upload an `.mp4` or `.mov` side-profile rowing
video. The app renders the first ten detected strokes into an annotated MP4,
then displays coaching feedback. Without `OPENAI_API_KEY`, it supplies a clear
metrics-based fallback instead of sending anything externally.

Reference strokes are loaded read-only from `../ten_strokes.json`; uploaded
videos and generated results stay under `web_app/data/` and are ignored by Git.
