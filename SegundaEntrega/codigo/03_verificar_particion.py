"""Comprueba que ningún caso aparezca en dos particiones: códigos, MD5 y pHash.

Salidas: resultados/verificacion_particion.txt y figuras/pares_phash.png
"""
import hashlib
from itertools import combinations

import imagehash
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import FIGURAS, RESULTADOS, cargar_config

cfg = cargar_config()
data_dir = cfg["data_dir"]
umbral = cfg["phash_threshold"]

part = pd.read_csv(RESULTADOS / "particion.csv")
lineas = []


def out(texto=""):
    print(texto)
    lineas.append(texto)


out("VERIFICACIÓN DE LA PARTICIÓN - Proyecto B (ORIGA-light)")
out("=" * 60)

# 1. Intersección de códigos
out("\n1. Intersección de códigos entre particiones")
grupos = {p: set(g["codigo"]) for p, g in part.groupby("particion")}
total_inter = 0
for a, b in combinations(sorted(grupos), 2):
    n = len(grupos[a] & grupos[b])
    total_inter += n
    out(f"   {a} ∩ {b}: {n}")
out(f"   Códigos repetidos dentro de particion.csv: {part['codigo'].duplicated().sum()}")
out(f"   Resultado: {'OK' if total_inter == 0 else 'FALLA'}")

# 2. Duplicados exactos por MD5
out("\n2. Duplicados exactos (MD5)")
part["md5"] = [hashlib.md5((data_dir / f).read_bytes()).hexdigest() for f in part["archivo"]]
dup = part[part["md5"].duplicated(keep=False)]
dentro = sum(g["md5"].duplicated().sum() for _, g in part.groupby("particion"))
entre = sum(g["particion"].nunique() > 1 for _, g in dup.groupby("md5"))
out(f"   Duplicados dentro de una misma partición: {dentro}")
out(f"   Grupos de MD5 repetidos entre particiones distintas: {entre}")
out(f"   Resultado: {'OK' if len(dup) == 0 else 'REVISAR'}")

# 3. Casi duplicados por pHash
out(f"\n3. Casi duplicados por pHash (imagehash.phash, 64 bits), umbral de Hamming <= {umbral}")
hashes = []
for f in part["archivo"]:
    with Image.open(data_dir / f) as im:
        hashes.append(imagehash.phash(im.convert("RGB")).hash.flatten())
H = np.array(hashes, dtype=bool)
D = (H[:, None, :] != H[None, :, :]).sum(axis=2)

particiones = part["particion"].to_numpy()
i, j = np.triu_indices(len(part), k=1)
distinta = particiones[i] != particiones[j]
pares = pd.DataFrame({
    "codigo_a": part["codigo"].to_numpy()[i[distinta]],
    "particion_a": particiones[i[distinta]],
    "codigo_b": part["codigo"].to_numpy()[j[distinta]],
    "particion_b": particiones[j[distinta]],
    "distancia": D[i[distinta], j[distinta]],
}).sort_values(["distancia", "codigo_a", "codigo_b"]).reset_index(drop=True)

out(f"   Pares comparados (de particiones distintas): {len(pares)}")
sospechosos = pares[pares["distancia"] <= umbral]
out(f"   Pares con distancia <= {umbral}: {len(sospechosos)}")
for r in sospechosos.itertuples():
    out(f"     {r.codigo_a} ({r.particion_a}) - {r.codigo_b} ({r.particion_b}): {r.distancia}")
out("   10 pares más cercanos (aunque superen el umbral):")
for r in pares.head(10).itertuples():
    out(f"     {r.codigo_a} ({r.particion_a}) - {r.codigo_b} ({r.particion_b}): {r.distancia}")
out(f"   Distancia mínima entre particiones: {pares['distancia'].min()} | mediana: {pares['distancia'].median():.0f}")
# Figura de pares para revisión visual: los sospechosos o, si no hay, los 5 más cercanos
a_mostrar = sospechosos if len(sospechosos) else pares.head(5)
titulo = ("Pares sospechosos (pHash <= %d)" % umbral) if len(sospechosos) else \
    "Sin pares bajo el umbral: 5 pares más cercanos entre particiones"
archivo_de = dict(zip(part["codigo"], part["archivo"]))
fig, ejes = plt.subplots(len(a_mostrar), 2, figsize=(5, 2.5 * len(a_mostrar)), squeeze=False)
for fila, r in zip(ejes, a_mostrar.itertuples()):
    for ax, cod, p in [(fila[0], r.codigo_a, r.particion_a), (fila[1], r.codigo_b, r.particion_b)]:
        with Image.open(data_dir / archivo_de[cod]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{cod} ({p}) | d={r.distancia}", fontsize=8)
        ax.axis("off")
fig.suptitle(titulo, fontsize=9)
fig.tight_layout()
fig.savefig(FIGURAS / "pares_phash.png", dpi=150)
out(f"   Figura de revisión: figuras/pares_phash.png ({len(a_mostrar)} pares)")

# 4. Conteos
out("\n4. Conteos por partición y etiqueta (1 = glaucoma, 0 = sano)")
tabla = pd.crosstab(part["particion"], part["etiqueta"], margins=True, margins_name="total")
tabla = tabla.reindex(["entrenamiento", "validacion", "prueba", "total"])
tabla["% glaucoma"] = (100 * tabla[1] / tabla["total"]).round(1)
out(tabla.to_string())

(RESULTADOS / "verificacion_particion.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
