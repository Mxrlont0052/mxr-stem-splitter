"""
separator.py
Clase MXRSeparator — encapsula los dos modelos de separación:
  · 2 stems → Mel-Band Roformer (vocals + instrumental)
  · 4 stems → Demucs htdemucs (vocals, drums, bass, other/melody)
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf
import yaml
from ml_collections import ConfigDict

from utils import demix_track, get_model_from_config


# ─── Rutas de modelos ─────────────────────────────────────────────────────────
# Descarga el checkpoint desde:
# https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt
# y ponlo en la carpeta models/

MEL_CONFIG  = Path("configs/config_vocals_mel_band_roformer.yaml")
MEL_CKPT    = Path("models/MelBandRoformer.ckpt")
DEMUCS_MODEL = "htdemucs"   # modelo 4-stem de Meta (se descarga automático)


class MXRSeparator:

    def __init__(self):
        self.ready = False
        self.mel_model = None
        self.mel_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ──────────────────────────────────────────────────────────────────────────
    # Carga de modelos
    # ──────────────────────────────────────────────────────────────────────────

    def load_models(self):
        """Carga ambos modelos al iniciar el servidor."""
        self._load_mel_band_roformer()
        # Demucs se descarga on-demand la primera vez que se usa
        self.ready = True

    def _load_mel_band_roformer(self):
        """Carga el modelo Mel-Band Roformer (2 stems: vocals + instrumental)."""
        if not MEL_CONFIG.exists():
            raise FileNotFoundError(
                f"Config no encontrado: {MEL_CONFIG}\n"
                "Asegúrate de tener configs/config_vocals_mel_band_roformer.yaml"
            )
        if not MEL_CKPT.exists():
            raise FileNotFoundError(
                f"Checkpoint no encontrado: {MEL_CKPT}\n"
                "Descarga el modelo desde:\n"
                "https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt\n"
                "y ponlo en models/MelBandRoformer.ckpt"
            )

        with open(MEL_CONFIG) as f:
            self.mel_config = ConfigDict(yaml.safe_load(f))

        self.mel_model = get_model_from_config("mel_band_roformer", self.mel_config)
        state = torch.load(str(MEL_CKPT), map_location="cpu")
        self.mel_model.load_state_dict(state)
        self.mel_model = self.mel_model.to(self.device)
        self.mel_model.eval()
        print(f"✅ Mel-Band Roformer cargado en {self.device.upper()}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2 STEMS — Mel-Band Roformer
    # ──────────────────────────────────────────────────────────────────────────

    def separate_2stem(self, input_path: str, output_dir: str) -> list[str]:
        """
        Separa en vocals + instrumental.
        Devuelve lista con los nombres de stems generados.
        """
        mix, sr = sf.read(input_path)

        # Mono → stereo
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        # Inferencia
        res, _ = demix_track(self.mel_config, self.mel_model, mixture, self.device)

        # Guardar vocals
        vocals = res["vocals"].T
        vocals_path = Path(output_dir) / "vocals.wav"
        sf.write(str(vocals_path), vocals, sr, subtype="FLOAT")

        # Guardar instrumental (mix original − vocals)
        instrumental = mix - vocals
        inst_path = Path(output_dir) / "instrumental.wav"
        sf.write(str(inst_path), instrumental, sr, subtype="FLOAT")

        print(f"✅ 2 stems guardados en {output_dir}")
        return ["vocals", "instrumental"]

    # ──────────────────────────────────────────────────────────────────────────
    # 4 STEMS — Demucs
    # ──────────────────────────────────────────────────────────────────────────

    def separate_4stem(self, input_path: str, output_dir: str) -> list[str]:
        """
        Separa en vocals, drums, bass, melody (other).
        Usa Demucs htdemucs vía subprocess.
        Devuelve lista con los nombres de stems generados.
        """
        # Demucs guarda los resultados en una subcarpeta con el nombre del modelo
        # Usamos un directorio temporal y luego movemos los archivos

        tmp_dir = Path(output_dir) / "_demucs_tmp"
        tmp_dir.mkdir(exist_ok=True)

        cmd = [
            "python", "-m", "demucs",
            "--name", DEMUCS_MODEL,
            "--out", str(tmp_dir),
            "--filename", "{stem}.wav",   # nombre limpio sin prefijos
            input_path
        ]

        print(f"⚙️  Corriendo Demucs: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Demucs falló:\n{result.stderr}"
            )

        # Demucs guarda en: tmp_dir/htdemucs/<nombre_archivo>/{vocals,drums,bass,other}.wav
        input_name = Path(input_path).stem
        demucs_out = tmp_dir / DEMUCS_MODEL / input_name

        stem_map = {
            "vocals": "vocals",
            "drums":  "drums",
            "bass":   "bass",
            "other":  "melody",   # "other" lo renombramos a "melody" para el frontend
        }

        generated = []
        for demucs_name, mxr_name in stem_map.items():
            src = demucs_out / f"{demucs_name}.wav"
            dst = Path(output_dir) / f"{mxr_name}.wav"
            if src.exists():
                shutil.move(str(src), str(dst))
                generated.append(mxr_name)
            else:
                print(f"⚠️  Stem no encontrado: {src}")

        # Limpiar carpeta temporal de demucs
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

        print(f"✅ 4 stems guardados en {output_dir}")
        return generated
