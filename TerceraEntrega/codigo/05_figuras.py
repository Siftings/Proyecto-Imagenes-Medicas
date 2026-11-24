"""Figuras de la primera ejecución: casos ilustrativos, matriz de confusión y curva ROC.

Salidas: figuras/casos_validacion.png, figuras/matriz_confusion.png, figuras/curva_roc.png
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import FIGURAS, RESULTADOS, cargar_config

cfg = cargar_config()
data_dir = cfg["data_dir"]

pred = pd.read_csv(RESULTADOS / "predicciones_validacion.csv")
part = pd.read_csv(RESULTADOS / "particion.csv")
archivo_de = dict(zip(part["codigo"], part["archivo"]))
with open(RESULTADOS / "metricas_validacion.json", encoding="utf-8") as f:
    met = json.load(f)
nombre = {0: "sano", 1: "glaucoma"}

# 1. Casos: un VP, un FN y un FP (VN si falta alguno). Se elige el más "típico" de cada tipo:
#    VP con mayor probabilidad, FN y FP con la probabilidad más alejada del umbral.
def elegir(tipo):
    sub = pred[pred["tipo"] == tipo]
    if sub.empty:
        return None
    clave = {"VP": -sub["prob_glaucoma"], "FN": sub["prob_glaucoma"],
             "FP": -sub["prob_glaucoma"], "VN": sub["prob_glaucoma"]}[tipo]
    return sub.loc[clave.sort_values().index[0]]

casos = [elegir(t) for t in ["VP", "FN", "FP"]]
reemplazo = iter(pred[pred["tipo"] == "VN"].sort_values("prob_glaucoma").itertuples())
casos = [c if c is not None else pd.Series(next(reemplazo)._asdict()) for c in casos]

fig, ejes = plt.subplots(1, 3, figsize=(10, 3.8))
for ax, c in zip(ejes, casos):
    with Image.open(data_dir / archivo_de[c["codigo"]]) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(f"{c['codigo']} | ref: {nombre[int(c['etiqueta'])]} | p={c['prob_glaucoma']:.2f} | "
                 f"pred: {nombre[int(c['prediccion'])]}", fontsize=8)
    ax.set_xlabel(c["tipo"], fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIGURAS / "casos_validacion.png", dpi=150)
print("Casos ilustrativos:", [(c["codigo"], c["tipo"], round(float(c["prob_glaucoma"]), 3)) for c in casos])

# 2. Matriz de confusión con conteos
M = np.array([[met["VP"], met["FN"]], [met["FP"], met["VN"]]])
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.imshow(M, cmap="Blues")
for (i, j), v in np.ndenumerate(M):
    etiqueta = [["VP", "FN"], ["FP", "VN"]][i][j]
    ax.text(j, i, f"{etiqueta}\n{v}", ha="center", va="center",
            color="white" if v > M.max() / 2 else "black", fontsize=12)
ax.set_xticks([0, 1], ["glaucoma", "sano"]); ax.set_yticks([0, 1], ["glaucoma", "sano"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Referencia")
ax.set_title(f"Validación (n={met['n']}), umbral {met['umbral']}", fontsize=10)
fig.tight_layout()
fig.savefig(FIGURAS / "matriz_confusion.png", dpi=150)

# 3. Curva ROC
fpr, tpr, _ = roc_curve(pred["etiqueta"], pred["prob_glaucoma"])
auc = met["metricas"]["AUC"]
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fpr, tpr, lw=2, label=f"Línea base (AUC = {auc['valor']:.3f}, IC95 {auc['IC95'][0]:.2f}-{auc['IC95'][1]:.2f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC = 0,5)")
sens, esp = met["metricas"]["sensibilidad"]["valor"], met["metricas"]["especificidad"]["valor"]
ax.plot(1 - esp, sens, "o", color="red", label=f"Umbral {met['umbral']}")
ax.set_xlabel("1 - especificidad"); ax.set_ylabel("Sensibilidad")
ax.set_title(f"Curva ROC en validación (n={met['n']})", fontsize=10)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIGURAS / "curva_roc.png", dpi=150)
print("Figuras guardadas en figuras/")
