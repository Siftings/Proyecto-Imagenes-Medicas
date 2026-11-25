"""Piezas gráficas de la presentación, hechas con las imágenes y los resultados del proyecto.

No entrena ni vuelve a evaluar la prueba: lee codigo/resultados/ y, para dos casos ya analizados en el
informe (test/580 y test/183), recalcula el mapa Grad-CAM con los pesos congelados. Solo escribe en img/
y en fragmentos.json (trozos de HTML/SVG que se pegan en index.html).
Uso, desde TerceraEntrega/presentacion/:  ../.venv/bin/python generar_piezas.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import roc_curve

AQUI = Path(__file__).resolve().parent
COD = AQUI.parent / "codigo"
os.chdir(COD)                       # config.yaml usa rutas relativas a codigo/
sys.path.insert(0, str(COD))
from comun import RESULTADOS, cargar_config, extraer_color, fijar_semillas  # noqa: E402
from modelos import Version, abrir, rasgos_cnn  # noqa: E402

IMG = AQUI / "img"
IMG.mkdir(exist_ok=True)
cfg = cargar_config()
fijar_semillas(cfg["seed"])
part = pd.read_csv(RESULTADOS / "particion.csv").set_index("codigo")
ruta = lambda c: cfg["data_dir"] / part.loc[c, "archivo"]
nombre = lambda c: c.replace("/", "_")
frag = {}

# Paleta de la lámina: negro -> bermellón -> amarillo del disco
TOPES = np.array([[13, 10, 8], [122, 24, 16], [224, 78, 30], [250, 222, 140]], float)


def rampa(v):
    v = np.clip(v, 0, 1)
    x = v * (len(TOPES) - 1)
    i = np.minimum(x.astype(int), len(TOPES) - 2)
    return TOPES[i] + (TOPES[i + 1] - TOPES[i]) * (x - i)[..., None]


def retina(c, lado=900):
    im = abrir(ruta(c)).convert("RGB").resize((lado, lado), Image.LANCZOS)
    im.filter(ImageFilter.UnsharpMask(2, 80, 2)).save(IMG / f"{nombre(c)}.jpg", quality=88)


# 1. Retinas sueltas
for c in ["test/366", "test/262", "test/289", "test/229", "test/197", "test/480", "test/016",
          "test/580", "test/183", "train/233"]:
    retina(c)
pares = pd.read_csv(RESULTADOS / "revision_pares.csv")
pares = pares[(pares.conclusion == "mismo_ojo") & (pares.particion_a != pares.particion_b)]
for c in [*pares.codigo_a, *pares.codigo_b]:
    retina(c, 480)
print(pares.to_string())

# 2. Grad-CAM de los dos casos analizados
v2 = Version("v2", cfg)
for c in ["test/580", "test/183"]:
    cam = v2.gradcam(ruta(c))
    n = 900
    g = np.asarray(abrir(ruta(c)).convert("L").resize((n, n), Image.LANCZOS), float) / 255
    m = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((n, n), Image.BICUBIC), float)[..., None] / 255
    base = g[..., None] * np.array([150, 138, 120])
    Image.fromarray(np.clip(base * (1 - .85 * m) + rampa(m[..., 0]) * m * .95, 0, 255).astype("uint8")).save(
        IMG / f"cam_{nombre(c)}.jpg", quality=90)

# 3. Los 512 rasgos (estandarizados) y las 8 medidas de train/233 (validación)
z = v2.lr.named_steps["scaler"].transform(rasgos_cnn([str(ruta("train/233"))], cfg, v2.red))[0]
t = (np.clip(z, -2, 2) + 2) / 4
Image.fromarray(rampa(t.reshape(16, 32)).astype("uint8")).resize((32 * 24, 16 * 24), Image.NEAREST).save(IMG / "rasgos512.png")
frag["ocho"] = [round(x, 3) for x in extraer_color(ruta("train/233"), cfg)]

# 4. ROC en validación (base y V2) como trazos SVG en una caja de 1000 x 1000
val = pd.read_csv(RESULTADOS / "predicciones_validacion_todas.csv")
for v in ["base", "v2"]:
    d = val[val.version == v]
    f, s, _ = roc_curve(d.etiqueta, d.puntuacion, drop_intermediate=False)
    frag["roc_" + v] = "M" + " L".join(f"{1000 * a:.1f},{1000 * (1 - b):.1f}" for a, b in zip(f, s))
d = val[val.version == "v2"]
frag["umbral_val"] = [int(((d.puntuacion >= .4834) & (d.etiqueta == 1)).sum()), int(((d.puntuacion < .4834) & (d.etiqueta == 0)).sum())]

# 5. Las 192 retinas de prueba en un mosaico: 20 por fila, un bloque por tipo de resultado
p = pd.read_csv(RESULTADOS / "predicciones_prueba.csv")
p = p[p.incluido_en_evaluacion]
L = 96
for tipo in ["VP", "FP", "FN", "VN"]:
    g = p[p.tipo == tipo].sort_values("puntuacion", ascending=False)
    filas = -(-len(g) // 20)
    h = Image.new("RGB", (20 * L, filas * L), (13, 10, 8))
    for k, c in enumerate(g.codigo):
        h.paste(abrir(ruta(c)).convert("RGB").resize((L, L), Image.LANCZOS), ((k % 20) * L, (k // 20) * L))
    h.save(IMG / f"mosaico_{tipo}.jpg", quality=86)
    frag["n_" + tipo] = len(g)
(AQUI / "fragmentos.json").write_text(json.dumps(frag, indent=1), encoding="utf-8")
print({k: (v if len(str(v)) < 90 else str(v)[:60]) for k, v in frag.items()})
