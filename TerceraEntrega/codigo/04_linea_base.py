"""Línea base: 8 características de color/palidez en la región del disco + regresión logística balanceada.

Solo usa entrenamiento (ajuste) y validación (una evaluación). La prueba no se toca.
Salidas: resultados/caracteristicas.csv, resultados/predicciones_validacion.csv,
         resultados/metricas_validacion.json, figuras/caracteristicas_por_clase.png,
         modelos/modelo_base.joblib (para evaluar en prueba si resulta elegida)
"""
import json
import platform

import joblib
import matplotlib
import numpy as np
import pandas as pd
import PIL
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import (CARACTERISTICAS, FIGURAS, MODELOS, RESULTADOS, bootstrap, cargar_config,
                   extraer_color, metricas)

cfg = cargar_config()
data_dir = cfg["data_dir"]
seed = cfg["seed"]
np.random.seed(seed)

# Características de entrenamiento y validación (nunca prueba)
part = pd.read_csv(RESULTADOS / "particion.csv")
part = part[part["particion"].isin(["entrenamiento", "validacion"])].reset_index(drop=True)
X_all = np.array([extraer_color(data_dir / f, cfg) for f in part["archivo"]])
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
joblib.dump(modelo, MODELOS / "modelo_base.joblib")

prob = modelo.predict_proba(X_val)[:, 1]
pred = (prob >= cfg["umbral"]).astype(int)

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
