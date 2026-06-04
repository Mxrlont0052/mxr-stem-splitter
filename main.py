import os
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from separator import MXRSeparator

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MXR Stem Splitter API",
    description="API para separar stems de audio — by Marlont / MXR",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción cambia esto por tu dominio frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directorios de trabajo ────────────────────────────────────────────────────

UPLOAD_DIR  = Path("temp_uploads")
OUTPUT_DIR  = Path("temp_outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif"}
MAX_FILE_MB = 50

# ─── Modelos cargados al iniciar ──────────────────────────────────────────────

separator: Optional[MXRSeparator] = None

@app.on_event("startup")
async def load_models():
    global separator
    print("⚙️  Cargando modelos MXR...")
    separator = MXRSeparator()
    separator.load_models()
    print("✅ Modelos listos.")

# ─── Schemas ──────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str          # "pending" | "processing" | "done" | "error"
    stems: list[str] = []
    error: Optional[str] = None

# ─── Almacén en memoria de jobs (reemplazar con Redis en producción) ──────────

jobs: dict[str, JobStatus] = {}

# ─── Utilidades ───────────────────────────────────────────────────────────────

def validate_audio_file(file: UploadFile) -> None:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {ext}. Usa: {', '.join(ALLOWED_EXTENSIONS)}"
        )

def cleanup_job_files(job_id: str):
    """Borra archivos temporales de un job (llamar con BackgroundTasks)."""
    job_dir = OUTPUT_DIR / job_id
    input_file = UPLOAD_DIR / f"{job_id}.*"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    for f in UPLOAD_DIR.glob(f"{job_id}.*"):
        f.unlink(missing_ok=True)

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = Path("index.html")
    if index.exists():
        return HTMLResponse(content=index.read_text(), status_code=200)
    return HTMLResponse(content="<h1>MXR Stem Splitter API</h1><p>Frontend no encontrado.</p>")

@app.get("/api")
async def api_info():
    return {
        "name": "MXR Stem Splitter",
        "version": "1.0.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "status": "online"
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": separator is not None and separator.ready,
        "cuda": torch.cuda.is_available()
    }


@app.post("/split/2stem", response_model=JobStatus)
async def split_2stem(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Sube una canción y separa en 2 stems: vocals + instrumental.
    Usa Mel-Band Roformer (modelo propio de MXR).
    """
    validate_audio_file(file)

    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    output_dir  = OUTPUT_DIR / job_id
    output_dir.mkdir(exist_ok=True)

    # Guardar archivo subido
    with open(input_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Archivo muy grande. Máximo {MAX_FILE_MB}MB.")
        f.write(content)

    # Registrar job
    jobs[job_id] = JobStatus(job_id=job_id, status="processing")

    # Procesar
    try:
        stems = separator.separate_2stem(str(input_path), str(output_dir))
        jobs[job_id] = JobStatus(job_id=job_id, status="done", stems=stems)
    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = JobStatus(job_id=job_id, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Limpiar el archivo de entrada después de 10 min
    background_tasks.add_task(lambda: input_path.unlink(missing_ok=True))

    return jobs[job_id]


@app.post("/split/4stem", response_model=JobStatus)
async def split_4stem(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Sube una canción y separa en 4 stems: vocals, drums, bass, melody.
    Usa Demucs (Meta).
    """
    validate_audio_file(file)

    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    output_dir  = OUTPUT_DIR / job_id
    output_dir.mkdir(exist_ok=True)

    with open(input_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Archivo muy grande. Máximo {MAX_FILE_MB}MB.")
        f.write(content)

    jobs[job_id] = JobStatus(job_id=job_id, status="processing")

    try:
        stems = separator.separate_4stem(str(input_path), str(output_dir))
        jobs[job_id] = JobStatus(job_id=job_id, status="done", stems=stems)
    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = JobStatus(job_id=job_id, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(lambda: input_path.unlink(missing_ok=True))

    return jobs[job_id]


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Consulta el estado de un job de separación."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    return jobs[job_id]


@app.get("/download/{job_id}/{stem_name}")
async def download_stem(
    job_id: str,
    stem_name: str,
    background_tasks: BackgroundTasks
):
    """
    Descarga un stem procesado.
    stem_name: vocals | instrumental | drums | bass | melody
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    if jobs[job_id].status != "done":
        raise HTTPException(status_code=400, detail="El job todavía no terminó.")

    stem_path = OUTPUT_DIR / job_id / f"{stem_name}.wav"
    if not stem_path.exists():
        raise HTTPException(status_code=404, detail=f"Stem '{stem_name}' no encontrado.")

    return FileResponse(
        path=str(stem_path),
        media_type="audio/wav",
        filename=f"MXR_{stem_name}.wav"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str, background_tasks: BackgroundTasks):
    """Borra los archivos de un job del servidor."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    background_tasks.add_task(cleanup_job_files, job_id)
    del jobs[job_id]
    return {"deleted": job_id}
