from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_EXTENSIONS, DATA_DIR
from app.models import JobStatus
from app.pipeline import run_pipeline

app = FastAPI(title="Offline Transcriber MVP")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def _job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def set_job_status(job_id: str, status: str, message: str, segments: list[dict] | None = None) -> None:
    payload = JobStatus(job_id=job_id, status=status, message=message, segments=segments)
    _status_path(job_id).write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def process_job(job_id: str, input_path: Path) -> None:
    try:
        set_job_status(job_id, "processing", "Running offline transcription pipeline")
        segments = run_pipeline(input_path, _job_dir(job_id))
        set_job_status(job_id, "completed", "Done", segments=segments)
    except Exception as exc:
        set_job_status(job_id, "failed", str(exc))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("app/static/index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only mp3, wav, m4a files are supported")

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    with input_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    set_job_status(job_id, "queued", "Job queued")
    background_tasks.add_task(process_job, job_id, input_path)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    status_file = _status_path(job_id)
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(status_file.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/result.json")
def download_json(job_id: str):
    path = _job_dir(job_id) / "result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result JSON not ready")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/result.srt")
def download_srt(job_id: str):
    path = _job_dir(job_id) / "result.srt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result SRT not ready")
    return FileResponse(path)
