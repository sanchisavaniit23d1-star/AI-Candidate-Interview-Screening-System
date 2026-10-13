"""
FastAPI backend for the AI Interview Analysis system.

Run from the project folder (where the checkpoints and models/ live):
    uvicorn api:app --host 127.0.0.1 --port 8000

Endpoints
    POST   /api/videos          upload a video -> {id, status, cached}
    GET    /api/jobs            recent jobs
    GET    /api/jobs/{id}       status + progress
    GET    /api/results/{id}    finished timeline + summary (409 while running)
    DELETE /api/jobs/{id}       delete a job and its files
    GET    /api/health
Jobs run one at a time in a background thread (CPU inference takes minutes per video).
"""
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = os.environ.get("DATA_DIR", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
WORK_DIR = os.path.join(DATA_DIR, "work")
DB_PATH = os.path.join(DATA_DIR, "jobs.db")
ALLOWED = {".mp4", ".avi", ".mov", ".mkv"}
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

app = FastAPI(title="AI Interview Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

_db_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db_lock, db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, filename TEXT, sha256 TEXT, status TEXT, progress REAL,
                message TEXT, error TEXT, created_at REAL, finished_at REAL, result_json TEXT)"""
        )
        # A restart kills running work: mark leftovers as failed.
        c.execute(
            "UPDATE jobs SET status='failed', error='Server restarted during analysis' "
            "WHERE status IN ('queued','running')"
        )


init_db()


def update(job_id, **fields):
    keys = ", ".join(f"{k}=?" for k in fields)
    with _db_lock, db() as c:
        c.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), job_id))


def job_row(job_id):
    with _db_lock, db() as c:
        return c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def public(row):
    return {
        "id": row["id"], "filename": row["filename"], "status": row["status"],
        "progress": row["progress"], "message": row["message"], "error": row["error"],
        "created_at": row["created_at"], "finished_at": row["finished_at"],
    }


def summarize(result):
    timeline = result.get("timeline", [])
    if not timeline:
        return {}
    counts = Counter(t["fusion"]["emotion"] for t in timeline)
    n = len(timeline)
    mean = lambda name: round(sum(t["hr_scores"][name] for t in timeline) / n, 2)
    return {
        "windows": n,
        "dominant_emotion": counts.most_common(1)[0][0],
        "emotion_window_counts": dict(counts),
        "mean_scores": {k: mean(k) for k in ("composure", "stress", "engagement")},
    }


def run_job(job_id, video_path):
    try:
        update(job_id, status="running", progress=0.0, message="Starting")
        from interview_analyzer import analyze_interview  # imported lazily: heavy (torch)

        def cb(fraction, message):
            update(job_id, progress=round(float(fraction), 3), message=message)

        result = analyze_interview(video_path, work_dir=os.path.join(WORK_DIR, job_id), progress_cb=cb)
        update(job_id, status="done", progress=1.0, message="Done", finished_at=time.time(),
               result_json=json.dumps(result))
    except Exception as exc:  # report any failure to the client instead of crashing the worker
        update(job_id, status="failed", error=str(exc)[:500], message="Failed", finished_at=time.time())


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED))}")

    job_id = uuid.uuid4().hex[:12]
    path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    sha, size = hashlib.sha256(), 0
    with open(path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                out.close()
                os.remove(path)
                raise HTTPException(413, f"File too large (limit {MAX_BYTES // (1024 * 1024)} MB).")
            sha.update(chunk)
            out.write(chunk)
    if size == 0:
        os.remove(path)
        raise HTTPException(400, "Empty file.")

    digest = sha.hexdigest()
    with _db_lock, db() as c:  # reuse a finished analysis of the identical file
        done = c.execute("SELECT id FROM jobs WHERE sha256=? AND status='done' ORDER BY finished_at DESC", (digest,)).fetchone()
    if done:
        os.remove(path)
        return {"id": done["id"], "status": "done", "cached": True}

    with _db_lock, db() as c:
        c.execute(
            "INSERT INTO jobs (id, filename, sha256, status, progress, message, created_at) VALUES (?,?,?,?,?,?,?)",
            (job_id, os.path.basename(file.filename), digest, "queued", 0.0, "Queued", time.time()),
        )
    _executor.submit(run_job, job_id, path)
    return {"id": job_id, "status": "queued", "cached": False}


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
    with _db_lock, db() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (min(limit, 100),)).fetchall()
    return [public(r) for r in rows]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    row = job_row(job_id)
    if not row:
        raise HTTPException(404, "Job not found.")
    return public(row)


@app.get("/api/results/{job_id}")
def get_result(job_id: str):
    row = job_row(job_id)
    if not row:
        raise HTTPException(404, "Job not found.")
    if row["status"] == "failed":
        raise HTTPException(422, row["error"] or "Analysis failed.")
    if row["status"] != "done":
        raise HTTPException(409, f"Analysis is {row['status']}.")
    result = json.loads(row["result_json"])
    return {"job": public(row), "summary": summarize(result), "result": result}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    row = job_row(job_id)
    if not row:
        raise HTTPException(404, "Job not found.")
    if row["status"] in ("queued", "running"):
        raise HTTPException(409, "Cannot delete a job that is still running.")
    with _db_lock, db() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    shutil.rmtree(os.path.join(WORK_DIR, job_id), ignore_errors=True)
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(job_id):
            os.remove(os.path.join(UPLOAD_DIR, f))
    return {"deleted": job_id}
