"""Funciones compartidas por los scripts: configuración, rutas, métricas y tablas por caso.

Todas las versiones (línea base y variantes) se evalúan con estas mismas funciones, para que
las métricas y sus denominadores sean comparables.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
FIGURAS = BASE / "figuras"
MODELOS = BASE / "modelos"

METRICAS = ["sensibilidad", "especificidad", "AUC", "F1", "exactitud"]
NOMBRE_CLASE = {0: "sano", 1: "glaucoma"}
CARACTERISTICAS = ["media_R", "media_G", "media_B", "std_R", "std_G", "std_B", "palidez_G", "palidez_R"]


def cargar_config():
    with open(BASE / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (BASE / data_dir).resolve()
    cfg["data_dir"] = data_dir
    RESULTADOS.mkdir(exist_ok=True)
    FIGURAS.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)
    return cfg


def codigo_de(nombre):
    """'ORIGA/train/093.jpg' -> 'train/093'."""
    p = Path(nombre)
    return f"{p.parent.name}/{p.stem}"


def fijar_semillas(seed):
    """Semillas de random, numpy y (si está instalado) torch, con algoritmos deterministas."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


# ------------------------------------------------------------------ características de color
def _mascaras(cfg):
    n = cfg["img_size"]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    return r <= cfg["r_centro"], (r > cfg["r_centro"]) & (r <= cfg["r_disco"]), r <= cfg["r_disco"]


def extraer_color(ruta, cfg):
    """Las 8 características de la línea base (Entrega 2): color en el disco y palidez centro/anillo."""
    centro, anillo, disco = _mascaras(cfg)
    n = cfg["img_size"]
    with Image.open(ruta) as im:
        img = im.convert("RGB").resize((n, n), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float64) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    return [
        R[disco].mean(), G[disco].mean(), B[disco].mean(),
        R[disco].std(), G[disco].std(), B[disco].std(),
        G[centro].mean() / G[anillo].mean(),
        R[centro].mean() / R[anillo].mean(),
    ]


# ----------------------------------------------------------------------------- métricas
def metricas(y, p, s):
    """Métricas con glaucoma (1) como clase positiva. Denominador 0 -> NaN (no calculable)."""
    vn, fp, fn, vp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "sensibilidad": vp / (vp + fn) if vp + fn else float("nan"),
        "especificidad": vn / (vn + fp) if vn + fp else float("nan"),
        "AUC": roc_auc_score(y, s) if len(set(y)) == 2 else float("nan"),
        "F1": f1_score(y, p, zero_division=0),
        "exactitud": accuracy_score(y, p),
    }, (int(vp), int(fn), int(fp), int(vn))


def bootstrap(y, p, s, n_boot, semilla):
    """IC 95 % percentil con remuestreo estratificado por clase."""
    rng = np.random.default_rng(semilla)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    res = {k: [] for k in METRICAS}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        m, _ = metricas(y[idx], p[idx], s[idx])
        for k, v in m.items():
            res[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


def bootstrap_pareado(y, s_a, s_b, n_boot, semilla):
    """Diferencia de AUC (b - a) sobre los mismos casos: valor e IC 95 % por bootstrap estratificado."""
    rng = np.random.default_rng(semilla)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    difs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        difs.append(roc_auc_score(y[idx], s_b[idx]) - roc_auc_score(y[idx], s_a[idx]))
    difs = np.array(difs)
    return {"valor": float(roc_auc_score(y, s_b) - roc_auc_score(y, s_a)),
            "IC95": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
            "fraccion_remuestreos_b_mejor": float((difs > 0).mean())}


def umbral_por_especificidad(y, s, esp_min):
    """Menor umbral (punto medio entre puntuaciones) cuya especificidad en (y, s) sea >= esp_min.

    Con pred = s >= umbral, la especificidad crece con el umbral; el menor umbral que cumple
    la condición es el que conserva la mayor sensibilidad.
    """
    unicos = np.unique(s)
    for i, t in enumerate(unicos):
        p = (s >= t).astype(int)
        m, _ = metricas(y, p, s)
        if m["especificidad"] >= esp_min:
            return float(t if i == 0 else (t + unicos[i - 1]) / 2)
    return float(unicos[-1] + 1e-6)


def tipo_caso(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.select([(y == 1) & (p == 1), (y == 0) & (p == 0), (y == 0) & (p == 1)],
                     ["VP", "VN", "FP"], "FN")


def resumen(y, s, umbral, n_boot, semilla):
    """Diccionario con matriz, denominadores, métricas e IC 95 % para un umbral dado."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    p = (s >= umbral).astype(int)
    m, (vp, fn, fp, vn) = metricas(y, p, s)
    ic = bootstrap(y, p, s, n_boot, semilla)
    return {
        "n": int(len(y)), "n_glaucoma": int(y.sum()), "n_sano": int((y == 0).sum()),
        "umbral": float(umbral),
        "VP": vp, "FN": fn, "FP": fp, "VN": vn,
        "denominadores": {"sensibilidad": f"{vp}/{vp + fn}", "especificidad": f"{vn}/{vn + fp}"},
        "metricas": {k: {"valor": v, "IC95": ic[k]} for k, v in m.items()},
        "bootstrap": {"n_remuestreos": n_boot, "semilla": semilla,
                      "tipo": "estratificado por clase, percentil"},
    }


def tabla_por_caso(codigos, particion, version, y, s, umbral):
    """Tabla por caso en el formato de la guía: ID, partición, versión, etiqueta, puntuación, predicción, tipo."""
    import pandas as pd
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    p = (s >= umbral).astype(int)
    return pd.DataFrame({
        "codigo": list(codigos), "particion": particion, "version": version,
        "etiqueta": y, "puntuacion": np.round(s, 4), "umbral": round(float(umbral), 4),
        "prediccion": p, "tipo": tipo_caso(y, p),
    })


def _limpiar(o):
    """NaN -> None (JSON null, 'no calculable') y tipos numpy -> nativos."""
    if isinstance(o, dict):
        return {k: _limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    return o


def guardar_json(obj, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(_limpiar(obj), f, indent=2, ensure_ascii=False)


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
