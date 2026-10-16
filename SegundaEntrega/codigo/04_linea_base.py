"""Línea base: 8 características de color/palidez en la región del disco + regresión logística balanceada.

Solo usa entrenamiento (ajuste) y validación (una evaluación). La prueba no se toca.
Salidas: resultados/caracteristicas.csv, resultados/predicciones_validacion.csv,
         resultados/metricas_validacion.json, figuras/caracteristicas_por_clase.png
"""
import json
import platform

import matplotlib
import numpy as np
import pandas as pd
import PIL
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import FIGURAS, RESULTADOS, cargar_config

cfg = cargar_config()
data_dir = cfg["data_dir"]
seed = cfg["seed"]
np.random.seed(seed)

CARACTERISTICAS = ["media_R", "media_G", "media_B", "std_R", "std_G", "std_B", "palidez_G", "palidez_R"]

# Máscaras circulares centradas en la imagen redimensionada
n = cfg["img_size"]
c = n // 2
yy, xx = np.mgrid[0:n, 0:n]
r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
centro = r <= cfg["r_centro"]
anillo = (r > cfg["r_centro"]) & (r <= cfg["r_disco"])
disco = r <= cfg["r_disco"]


def extraer(ruta):
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


# Características de entrenamiento y validación (nunca prueba)
part = pd.read_csv(RESULTADOS / "particion.csv")
part = part[part["particion"].isin(["entrenamiento", "validacion"])].reset_index(drop=True)
X_all = np.array([extraer(data_dir / f) for f in part["archivo"]])
car = pd.concat([part[["codigo", "particion", "etiqueta"]],
                 pd.DataFrame(X_all, columns=CARACTERISTICAS)], axis=1)
car.to_csv(RESULTADOS / "caracteristicas.csv", index=False, float_format="%.6f")

ent = car[car["particion"] == "entrenamiento"]
val = car[car["particion"] == "validacion"]
X_ent, y_ent = ent[CARACTERISTICAS].to_numpy(), ent["etiqueta"].to_numpy()
X_val, y_val = val[CARACTERISTICAS].to_numpy(), val["etiqueta"].to_numpy()

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(class_weight="balanced", C=cfg["C"], max_iter=1000)),
])
modelo.fit(X_ent, y_ent)

prob = modelo.predict_proba(X_val)[:, 1]
pred = (prob >= cfg["umbral"]).astype(int)


def metricas(y, p, s):
    vn, fp, fn, vp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "sensibilidad": vp / (vp + fn) if vp + fn else float("nan"),
        "especificidad": vn / (vn + fp) if vn + fp else float("nan"),
        "AUC": roc_auc_score(y, s),
        "F1": f1_score(y, p, zero_division=0),
        "exactitud": accuracy_score(y, p),
    }, (int(vp), int(fn), int(fp), int(vn))


def bootstrap(y, p, s, n_boot, semilla):
    """IC 95 % percentil con remuestreo estratificado por clase."""
    rng = np.random.default_rng(semilla)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    res = {k: [] for k in ["sensibilidad", "especificidad", "AUC", "F1", "exactitud"]}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        m, _ = metricas(y[idx], p[idx], s[idx])
        for k, v in m.items():
            res[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


m, (vp, fn, fp, vn) = metricas(y_val, pred, prob)
ic = bootstrap(y_val, pred, prob, cfg["n_bootstrap"], seed)

# Referencia trivial: siempre "sano"
pred_triv = np.zeros_like(y_val)
m_triv, (tvp, tfn, tfp, tvn) = metricas(y_val, pred_triv, np.zeros(len(y_val)))

# AUC en entrenamiento, para vigilar sobreajuste
auc_ent = roc_auc_score(y_ent, modelo.predict_proba(X_ent)[:, 1])

tipo = np.select([(y_val == 1) & (pred == 1), (y_val == 0) & (pred == 0), (y_val == 0) & (pred == 1)],
                 ["VP", "VN", "FP"], "FN")
pd.DataFrame({
    "codigo": val["codigo"].to_numpy(), "etiqueta": y_val,
    "prob_glaucoma": np.round(prob, 4), "prediccion": pred, "tipo": tipo,
}).to_csv(RESULTADOS / "predicciones_validacion.csv", index=False)

coef = modelo.named_steps["lr"].coef_[0]
salida = {
    "n": int(len(y_val)), "n_glaucoma": int(y_val.sum()), "n_sano": int((y_val == 0).sum()),
    "umbral": cfg["umbral"],
    "VP": vp, "FN": fn, "FP": fp, "VN": vn,
    "metricas": {k: {"valor": float(v), "IC95": ic[k]} for k, v in m.items()},
    "bootstrap": {"n_remuestreos": cfg["n_bootstrap"], "semilla": seed, "tipo": "estratificado por clase, percentil"},
    "clasificador_trivial_siempre_sano": {"VP": tvp, "FN": tfn, "FP": tfp, "VN": tvn,
                                         **{k: float(v) for k, v in m_triv.items()}},
    "AUC_entrenamiento": float(auc_ent),
    "coeficientes_estandarizados": {k: float(v) for k, v in zip(CARACTERISTICAS, coef)},
    "intercepto": float(modelo.named_steps["lr"].intercept_[0]),
    "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                  "pillow": PIL.__version__, "sklearn": sklearn.__version__},
}
with open(RESULTADOS / "metricas_validacion.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, indent=2, ensure_ascii=False)

# Boxplots de palidez por clase (solo entrenamiento)
fig, ejes = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, col in zip(ejes, ["palidez_G", "palidez_R"]):
    ax.boxplot([ent.loc[ent["etiqueta"] == 0, col], ent.loc[ent["etiqueta"] == 1, col]])
    ax.set_xticks([1, 2], [f"Sano (n={(y_ent == 0).sum()})", f"Glaucoma (n={(y_ent == 1).sum()})"])
    ax.set_title(col)
fig.suptitle("Palidez central por clase (solo entrenamiento)", fontsize=10)
fig.tight_layout()
fig.savefig(FIGURAS / "caracteristicas_por_clase.png", dpi=150)

# Resumen
print(f"Validación n={len(y_val)} | VP={vp} FN={fn} FP={fp} VN={vn}")
for k, v in m.items():
    print(f"  {k:14s} {v:.3f}  IC95 [{ic[k][0]:.3f}, {ic[k][1]:.3f}]")
print(f"Trivial (siempre sano): " + ", ".join(f"{k}={v:.3f}" for k, v in m_triv.items()))
print(f"AUC entrenamiento: {auc_ent:.3f}")
print("Coeficientes:", {k: round(float(v), 3) for k, v in zip(CARACTERISTICAS, coef)})
