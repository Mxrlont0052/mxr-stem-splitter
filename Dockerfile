FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models', exist_ok=True); hf_hub_download(repo_id='KimberleyJSN/melbandroformer', filename='MelBandRoformer.ckpt', local_dir='models')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
