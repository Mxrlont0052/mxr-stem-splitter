# MXR Stem Splitter — Backend

API en Python (FastAPI) para separar stems de audio.
**by Marlont / MXR**

---

## Estructura del proyecto

```
mxr-backend/
├── main.py                          ← API FastAPI (endpoints)
├── separator.py                     ← Lógica de separación (modelos)
├── utils.py                         ← Del repo original (demix_track, etc.)
├── requirements.txt
├── configs/
│   └── config_vocals_mel_band_roformer.yaml
├── models/
│   ├── mel_band_roformer/           ← Código del modelo (del zip original)
│   │   ├── __init__.py
│   │   ├── attend.py
│   │   └── mel_band_roformer.py
│   └── MelBandRoformer.ckpt         ← ⬇️ DEBES DESCARGAR ESTO (ver abajo)
├── temp_uploads/                    ← Archivos subidos (auto-limpieza)
└── temp_outputs/                    ← Stems generados (auto-limpieza)
```

---

## Setup paso a paso

### 1. Clonar / copiar archivos

Copia todo este proyecto a tu servidor o máquina local.

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Si usas GPU asegúrate de instalar la versión correcta de PyTorch para tu CUDA:
> https://pytorch.org/get-started/locally/

### 3. Descargar el checkpoint del modelo

```bash
# Opción A — manual
# Ve a: https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt
# Descarga el archivo y ponlo en: models/MelBandRoformer.ckpt

# Opción B — desde terminal
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='KimberleyJSN/melbandroformer',
    filename='MelBandRoformer.ckpt',
    local_dir='models'
)
print('Descargado en:', path)
"
```

### 4. Copiar archivos del modelo original (del zip que tienes)

Del zip `Mel-Band-Roformer-Vocal-Model-main.zip` copia:

```
models/mel_band_roformer/__init__.py       → models/mel_band_roformer/__init__.py
models/mel_band_roformer/attend.py         → models/mel_band_roformer/attend.py
models/mel_band_roformer/mel_band_roformer.py → models/mel_band_roformer/mel_band_roformer.py
utils.py                                   → utils.py
configs/config_vocals_mel_band_roformer.yaml → configs/config_vocals_mel_band_roformer.yaml
```

### 5. Levantar el servidor

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

El servidor queda en: `http://localhost:8000`
Documentación automática: `http://localhost:8000/docs`

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Info del servidor |
| GET | `/health` | Estado y modelos cargados |
| POST | `/split/2stem` | Separar vocals + instrumental |
| POST | `/split/4stem` | Separar vocals, drums, bass, melody |
| GET | `/job/{job_id}` | Estado de un job |
| GET | `/download/{job_id}/{stem}` | Descargar un stem |
| DELETE | `/job/{job_id}` | Borrar archivos del job |

---

## Ejemplo de uso (curl)

```bash
# Separar en 2 stems
curl -X POST http://localhost:8000/split/2stem \
  -F "file=@mi_cancion.wav" \
  | python -m json.tool

# Respuesta:
# {
#   "job_id": "abc-123-...",
#   "status": "done",
#   "stems": ["vocals", "instrumental"]
# }

# Descargar stem
curl -O http://localhost:8000/download/abc-123.../vocals
```

---

## Deploy en Railway (gratis)

1. Crea cuenta en https://railway.app
2. Conecta tu repo de GitHub con este proyecto
3. Railway detecta Python automáticamente
4. Agrega variable de entorno si necesitas: `PORT=8000`
5. Listo — te da una URL pública para conectar con el frontend

> ⚠️ Railway free tier no tiene GPU. Para GPU usa RunPod o vast.ai.

---

## Variables de entorno (opcional)

```env
MAX_FILE_MB=50        # Tamaño máximo de archivo en MB
CORS_ORIGINS=*        # Dominios permitidos (en prod, pon tu dominio frontend)
```
