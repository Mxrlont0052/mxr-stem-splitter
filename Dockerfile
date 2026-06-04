FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url 
https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('models', exist_ok=True); \
hf_hub_download(repo_id='KimberleyJSN/melbandroformer', 
filename='MelBandRoformer.ckpt', local_dir='models'); \
print('Modelo descargado OK')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]≈# ── 
Stage 1: 
base con dependencias del sistema 
─────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: instalar dependencias Python ──────────────────────────────────
COPY requirements.txt .

# Torch CPU-only primero (más liviano, Railway no tiene GPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# ── Stage 3: descargar modelos en build time ────────────────────────────────
# Mel-Band Roformer checkpoint (~400 MB)
RUN mkdir -p models && \
    wget -q --show-progress \
    "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt" \
    -O models/MelBandRoformer.ckpt

# Pre-caché del modelo Demucs htdemucs (~80 MB)
RUN python -c "import demucs.pretrained; demucs.pretrained.get_model('htdemucs')"

# ── Copiar código ───────────────────────────────────────────────────────────
COPY . .

# Crear directorios de trabajo (el filesystem de Railway es efímero, está bien)
RUN mkdir -p temp_uploads temp_outputs

# ── Arranque ────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
