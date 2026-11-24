"""Figuras de la entrega final. Cada figura indica método, partición y número de casos.

Validación (n=91): roc_comparacion_validacion.png, errores_exposicion_validacion.png, historial_v3.png
Prueba (versión elegida): matriz_confusion_prueba.png, roc_prueba.png, casos_prueba.png,
                          casos_analizados_prueba.png (error + caso límite, con Grad-CAM)
Lee solo archivos de resultados/ y los pesos congelados; no recalcula métricas ni reajusta nada.
"""
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import FIGURAS, NOMBRE_CLASE, RESULTADOS, cargar_config, fijar_semillas, leer_json
from modelos import DESCRIPCION, Version

cfg = cargar_config()
fijar_semillas(cfg["seed"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.grid": True, "grid.color": "#e6e6e3",
                     "grid.linewidth": 0.6})
# Paleta categórica validada (orden fijo) + estilo de línea como segunda codificación
COLOR = {"base": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a", "v3": "#eda100"}
ESTILO = {"base": "-", "v1": "--", "v2": "-", "v3": ":"}
ETIQ = {"base": "Base (color)", "v1": "V1 (CNN+LR C=1)", "v2": "V2 (CNN+LR C-CV)", "v3": "V3 (ajuste fino)"}
TINTA = "#2b2b29"

part = pd.read_csv(RESULTADOS / "particion.csv").set_index("codigo")
comp = pd.read_csv(RESULTADOS / "comparacion_validacion.csv").set_index("version")
todas = pd.read_csv(RESULTADOS / "predicciones_validacion_todas.csv")
eleg = leer_json(RESULTADOS / "version_elegida.json")
v_eleg, umbral = eleg["version"], eleg["umbral"]


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIGURAS / nombre, dpi=200)
    plt.close(fig)


def imagen(codigo):
    with Image.open(cfg["data_dir"] / part.loc[codigo, "archivo"]) as im:
        return np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR))


# 1. ROC comparativa en validación
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for v in comp.index:
    t = todas[todas["version"] == v]
    fpr, tpr, _ = roc_curve(t["etiqueta"], t["puntuacion"])
    ax.plot(fpr, tpr, ESTILO[v], color=COLOR[v], lw=2,
            label=f"{ETIQ[v]}: AUC {comp.loc[v, 'AUC']:.3f} ({comp.loc[v, 'AUC_IC95'].replace('-', '–')})")
ax.plot([0, 1], [0, 1], color="#b5b5b0", lw=1)
ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad")
ax.set_title(f"ROC en validación, mismas {int(comp['n'].iloc[0])} imágenes", fontsize=9, color=TINTA)
ax.legend(fontsize=7, loc="lower right", frameon=False)
ax.set_aspect("equal")
guardar(fig, "roc_comparacion_validacion.png")

# 2. Error por tercil de brillo en validación (umbral 0,5)
car = pd.read_csv(RESULTADOS / "caracteristicas.csv").set_index("codigo")
cods = todas[todas["version"] == "base"]["codigo"].to_numpy()
brillo = (0.299 * car["media_R"] + 0.587 * car["media_G"] + 0.114 * car["media_B"]).loc[cods].to_numpy()
tercil = pd.Series(pd.qcut(brillo, 3, labels=["oscura", "media", "clara"]), index=cods)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ancho = 0.2
for i, v in enumerate(comp.index):
    t = todas[todas["version"] == v].set_index("codigo")
    err = (t["prediccion"] != t["etiqueta"])
    tasas = [err[tercil == g].mean() * 100 for g in ["oscura", "media", "clara"]]
    ns = [(tercil == g).sum() for g in ["oscura", "media", "clara"]]
    ax.bar(np.arange(3) + (i - 1.5) * ancho, tasas, ancho - 0.02, color=COLOR[v], label=ETIQ[v])
ax.set_xticks(range(3), [f"{g}\n(n={n})" for g, n in zip(["oscura", "media", "clara"], ns)])
ax.set_ylabel("Errores (%)"); ax.set_xlabel("Tercil de brillo del disco")
ax.set_title(f"Tasa de error por exposición, validación (n={len(cods)}), umbral 0,5", fontsize=9, color=TINTA)
ax.legend(fontsize=7, ncol=2, frameon=False)
ax.grid(axis="x", visible=False)
guardar(fig, "errores_exposicion_validacion.png")

# 3. Curva de pérdida de V3 (solo entrenamiento)
h = RESULTADOS / "historial_v3.csv"
if h.exists():
    h = pd.read_csv(h)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(h["epoca"], h["perdida_entrenamiento"], color=COLOR["v3"], lw=2, marker="o", ms=3)
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (BCE ponderada)")
    ax.set_title("V3: pérdida en entrenamiento (n=363)", fontsize=9, color=TINTA)
    guardar(fig, "historial_v3.png")

# --- Prueba: solo existe tras 11_evaluacion_prueba.py
if not (RESULTADOS / "metricas_prueba.json").exists():
    print("Sin evaluación de prueba todavía: solo figuras de validación.")
    raise SystemExit
met = leer_json(RESULTADOS / "metricas_prueba.json")["principal_192"]
pp = pd.read_csv(RESULTADOS / "predicciones_prueba.csv")
inc = pp[pp["incluido_en_evaluacion"]]
rot = f"{ETIQ[v_eleg]}, prueba (n={met['n']})"

# 4. Matriz de confusión en prueba
M = np.array([[met["VP"], met["FN"]], [met["FP"], met["VN"]]])
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(M, cmap="Blues", vmin=0)
for (i, j), v in np.ndenumerate(M):
    ax.text(j, i, f"{[['VP', 'FN'], ['FP', 'VN']][i][j]}\n{v}", ha="center", va="center",
            color="white" if v > M.max() / 2 else TINTA, fontsize=12)
ax.set_xticks([0, 1], ["glaucoma", "sano"]); ax.set_yticks([0, 1], ["glaucoma", "sano"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Referencia"); ax.grid(False)
ax.set_title(f"{rot}\numbral {umbral:.3f}", fontsize=8, color=TINTA)
guardar(fig, "matriz_confusion_prueba.png")

# 5. ROC en prueba con el punto de operación congelado
fpr, tpr, _ = roc_curve(inc["etiqueta"], inc["puntuacion"])
A = met["metricas"]["AUC"]
fig, ax = plt.subplots(figsize=(3.8, 3.6))
ax.plot(fpr, tpr, color=COLOR[v_eleg], lw=2, label=f"AUC {A['valor']:.3f} ({A['IC95'][0]:.2f}–{A['IC95'][1]:.2f})")
ax.plot([0, 1], [0, 1], color="#b5b5b0", lw=1)
s, e = met["metricas"]["sensibilidad"]["valor"], met["metricas"]["especificidad"]["valor"]
ax.plot(1 - e, s, "o", ms=8, color=TINTA, label=f"umbral {umbral:.3f}: sens {met['denominadores']['sensibilidad']}, "
                                                 f"esp {met['denominadores']['especificidad']}")
ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad")
ax.set_title(f"ROC, {rot}", fontsize=8, color=TINTA)
ax.legend(fontsize=7, loc="lower right", frameon=False); ax.set_aspect("equal")
guardar(fig, "roc_prueba.png")


# 6. Casos de prueba ilustrados: VP, VN, FP y FN más representativos (los más alejados del umbral)
def mas_lejano(tipo):
    sub = inc[inc["tipo"] == tipo]
    return None if sub.empty else sub.loc[(sub["puntuacion"] - umbral).abs().idxmax()]


casos = [c for c in (mas_lejano(t) for t in ["VP", "VN", "FP", "FN"]) if c is not None]
fig, ejes = plt.subplots(1, len(casos), figsize=(2.6 * len(casos), 3.0))
for ax, c in zip(np.atleast_1d(ejes), casos):
    ax.imshow(imagen(c["codigo"])); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"{c['codigo']} | ref: {NOMBRE_CLASE[int(c['etiqueta'])]}\np={c['puntuacion']:.2f} | "
                 f"pred: {NOMBRE_CLASE[int(c['prediccion'])]}", fontsize=8)
    ax.set_xlabel(c["tipo"], fontsize=10, fontweight="bold")
fig.suptitle(f"Casos ilustrativos: {rot}", fontsize=9)
guardar(fig, "casos_prueba.png")

# 7. Dos casos analizados de la versión final: el error más marcado y el caso más cercano al umbral
errores = inc[inc["tipo"].isin(["FN", "FP"])]
fn = inc[inc["tipo"] == "FN"]
error = (fn.loc[fn["puntuacion"].idxmin()] if not fn.empty else
         errores.loc[(errores["puntuacion"] - umbral).abs().idxmax()])
resto = inc[inc["codigo"] != error["codigo"]]
limite = resto.loc[(resto["puntuacion"] - umbral).abs().idxmin()]
version = Version(v_eleg, cfg)
fig, ejes = plt.subplots(2, 2, figsize=(5.4, 5.6))
for fila, (c, rol) in enumerate([(error, "Error observado"), (limite, "Caso límite")]):
    img = imagen(c["codigo"])
    cam = version.gradcam(cfg["data_dir"] / part.loc[c["codigo"], "archivo"])
    ejes[fila, 0].imshow(img)
    ejes[fila, 0].set_title(f"{rol}: {c['codigo']}\nref: {NOMBRE_CLASE[int(c['etiqueta'])]} | p={c['puntuacion']:.3f} | "
                            f"pred: {NOMBRE_CLASE[int(c['prediccion'])]} ({c['tipo']})", fontsize=8)
    ejes[fila, 1].imshow(img)
    if cam is not None:
        ejes[fila, 1].imshow(cam, cmap="inferno", alpha=0.45)
    ejes[fila, 1].set_title("Grad-CAM (layer4)" if cam is not None else "(sin mapa para la base)", fontsize=8)
    for ax in ejes[fila]:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle(f"Casos analizados: {rot}, umbral {umbral:.3f}", fontsize=9)
guardar(fig, "casos_analizados_prueba.png")
pd.DataFrame([{"rol": "error", **error.to_dict()}, {"rol": "limite", **limite.to_dict()}]).to_csv(
    RESULTADOS / "casos_analizados_prueba.csv", index=False)
print("Casos ilustrados:", [(c["codigo"], c["tipo"], round(float(c["puntuacion"]), 3)) for c in casos])
print("Casos analizados:", error["codigo"], error["tipo"], "|", limite["codigo"], limite["tipo"])
print("Figuras guardadas en figuras/")
