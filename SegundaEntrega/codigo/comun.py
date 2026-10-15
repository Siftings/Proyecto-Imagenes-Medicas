"""Funciones compartidas por los scripts: lectura de config y rutas."""
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
FIGURAS = BASE / "figuras"


def cargar_config():
    with open(BASE / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (BASE / data_dir).resolve()
    cfg["data_dir"] = data_dir
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)
    return cfg


def codigo_de(nombre):
    """'ORIGA/train/093.jpg' -> 'train/093'."""
    p = Path(nombre)
    return f"{p.parent.name}/{p.stem}"
