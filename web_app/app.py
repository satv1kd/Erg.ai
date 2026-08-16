from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analysis import analyse_video

ROOT = Path(__file__).resolve().parent
UPLOADS, RESULTS = ROOT / "data" / "uploads", ROOT / "data" / "results"
REFERENCE = ROOT.parent / "ten_strokes.json"
for folder in (UPLOADS, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Erg.ai Studio")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
jobs: dict[str, dict] = {}


def run_job(job_id: str, source: Path, destination: Path) -> None:
    try:
        jobs[job_id].update(status="processing", message="Tracking strokes and rendering your overlay…")
        outcome = analyse_video(source, destination, REFERENCE)
        jobs[job_id].update(status="complete", message="Analysis complete.", result=outcome)
    except Exception as exc:
        jobs[job_id].update(status="error", message=str(exc))


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/analyse")
async def upload_video(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(415, "Please upload an MP4 or MOV video.")
    if not REFERENCE.exists():
        raise HTTPException(500, "Reference stroke data (ten_strokes.json) is unavailable.")
    job_id = uuid.uuid4().hex
    source, destination = UPLOADS / f"{job_id}{suffix}", RESULTS / f"{job_id}.mp4"
    with source.open("wb") as file:
        shutil.copyfileobj(video.file, file)
    jobs[job_id] = {"status": "queued", "message": "Video uploaded. Preparing analysis…", "video": f"/api/jobs/{job_id}/video"}
    threading.Thread(target=run_job, args=(job_id, source, destination), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "Analysis job not found.")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/video")
def result_video(job_id: str) -> FileResponse:
    path = RESULTS / f"{job_id}.mp4"
    if job_id not in jobs or jobs[job_id]["status"] != "complete" or not path.exists():
        raise HTTPException(404, "Annotated video is not ready.")
    return FileResponse(path, media_type="video/mp4", filename="erg-ai-analysis.mp4")
