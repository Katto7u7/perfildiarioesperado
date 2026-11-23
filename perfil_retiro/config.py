"""Carga de configuración (YAML) y resolución de rutas."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]          # prod_perfil_retiro/


@dataclass(frozen=True)
class Config:
    raw: dict

    def __getitem__(self, k):
        return self.raw[k]

    def ruta(self, clave: str) -> Path:
        f = self.raw["fuentes"]
        if clave in ("db_visita", "tope_csv", "politica_dir"):
            return (RAIZ / f[clave]).resolve()
        return (RAIZ / f["db_dir"] / f[clave]).resolve()

    @property
    def hash(self) -> str:
        # `destino` (a dónde se escribe, Postgres on/off) no cambia los resultados: no entra en la huella,
        # para que mover el destino no genere ids nuevos ni reentrene modelos. (Cambio 2026-11-20.)
        sin_destino = {k: v for k, v in self.raw.items() if k != "destino"}
        return hashlib.sha256(json.dumps(sin_destino, sort_keys=True).encode()).hexdigest()[:8]


def cargar_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else RAIZ / "config" / "config.yaml"
    with open(p, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
