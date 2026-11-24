"""Comprueba que ningún caso aparezca en dos particiones: códigos, MD5 y pHash.

Salidas: resultados/verificacion_particion.txt, resultados/revision_pares.csv,
         resultados/prueba_excluidos.csv, figuras/pares_phash.png, figuras/pares_consecutivos.png
         y figuras/pares_mismo_ojo.png
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
etiquetas = part["etiqueta"].to_numpy()
misma = (~distinta) & (D[i, j] <= umbral)
out(f"   Pares dentro de una misma partición con distancia <= {umbral} (no son fuga, pero indican repeticiones): {misma.sum()}")
for a, b in zip(i[misma], j[misma]):
    alerta = " | ETIQUETAS DISTINTAS" if etiquetas[a] != etiquetas[b] else ""
    out(f"     {part['codigo'][a]} - {part['codigo'][b]} ({particiones[a]}, y={etiquetas[a]}/{etiquetas[b]}): {D[a, b]}{alerta}")

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

# 3b. Códigos consecutivos: en la revisión visual, las repeticiones del mismo ojo aparecieron
#     con números de imagen vecinos, a veces con distancia algo mayor que el umbral general.
umbral_c = cfg["phash_threshold_consecutivos"]
out(f"\n3b. Pares de códigos consecutivos (n, n+1) con pHash <= {umbral_c}, en cualquier partición")
num = part["codigo"].str[-3:].astype(int).to_numpy()
orden = np.argsort(num)
consec = []
for a, b in zip(orden[:-1], orden[1:]):
    if num[b] == num[a] + 1 and D[a, b] <= umbral_c:
        consec.append((part["codigo"][a], particiones[a], part["etiqueta"][a],
                       part["codigo"][b], particiones[b], part["etiqueta"][b], D[a, b]))
out(f"   Pares encontrados: {len(consec)} (de {len(part) - 1} pares consecutivos)")
for ca, pa, ya, cb, pb, yb, d in consec:
    marca = "MISMA PARTICIÓN" if pa == pb else "ENTRE PARTICIONES"
    alerta = " | ETIQUETAS DISTINTAS" if ya != yb else ""
    out(f"     {ca} ({pa}, y={ya}) - {cb} ({pb}, y={yb}): {d} | {marca}{alerta}")
if consec:
    fig, ejes = plt.subplots(len(consec), 2, figsize=(5, 2.5 * len(consec)), squeeze=False)
    for fila, (ca, pa, ya, cb, pb, yb, d) in zip(ejes, consec):
        for ax, cod, p, y in [(fila[0], ca, pa, ya), (fila[1], cb, pb, yb)]:
            with Image.open(data_dir / archivo_de[cod]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{cod} ({p}, y={y}) | d={d}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Códigos consecutivos con pHash <= {umbral_c}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURAS / "pares_consecutivos.png", dpi=150)
    out("   Figura de revisión: figuras/pares_consecutivos.png")
out("   Estos pares deben revisarse a ojo: la conclusión de la revisión se registra en el informe.")

# 4. Conteos
out("\n4. Conteos por partición y etiqueta (1 = glaucoma, 0 = sano)")
tabla = pd.crosstab(part["particion"], part["etiqueta"], margins=True, margins_name="total")
tabla = tabla.reindex(["entrenamiento", "validacion", "prueba", "total"])
tabla["% glaucoma"] = (100 * tabla[1] / tabla["total"]).round(1)
out(tabla.to_string())

# 5. Conclusión de la revisión visual (registrada en config.yaml) y exclusiones de prueba
out("\n5. Revisión visual (config.yaml -> revision_visual)")
info = part.set_index("codigo")
rev = cfg.get("revision_visual", {})
filas_rev = []
for conclusion, lista in rev.items():
    for a, b in lista:
        ia, ib = part.index[part["codigo"] == a][0], part.index[part["codigo"] == b][0]
        filas_rev.append({
            "codigo_a": a, "particion_a": info.loc[a, "particion"], "etiqueta_a": int(info.loc[a, "etiqueta"]),
            "codigo_b": b, "particion_b": info.loc[b, "particion"], "etiqueta_b": int(info.loc[b, "etiqueta"]),
            "distancia_phash": int(D[ia, ib]), "conclusion": conclusion,
        })
rev_df = pd.DataFrame(filas_rev)
rev_df.to_csv(RESULTADOS / "revision_pares.csv", index=False)
for r in rev_df.itertuples():
    alerta = " | ETIQUETAS DISTINTAS" if r.etiqueta_a != r.etiqueta_b else ""
    out(f"   {r.conclusion:12s} {r.codigo_a} ({r.particion_a}) - {r.codigo_b} ({r.particion_b}): d={r.distancia_phash}{alerta}")

mismo = rev_df[rev_df["conclusion"] == "mismo_ojo"]
excluir = []
for r in mismo.itertuples():
    for cod, p, gem, pg in [(r.codigo_a, r.particion_a, r.codigo_b, r.particion_b),
                            (r.codigo_b, r.particion_b, r.codigo_a, r.particion_a)]:
        if p == "prueba" and pg != "prueba":
            excluir.append({"codigo": cod, "etiqueta": int(info.loc[cod, "etiqueta"]), "gemelo": gem,
                            "particion_gemelo": pg, "distancia_phash": r.distancia_phash})
exc = pd.DataFrame(excluir, columns=["codigo", "etiqueta", "gemelo", "particion_gemelo", "distancia_phash"])
exc.to_csv(RESULTADOS / "prueba_excluidos.csv", index=False)
fuga_val = mismo[((mismo["particion_a"] == "validacion") & (mismo["particion_b"] == "entrenamiento")) |
                 ((mismo["particion_a"] == "entrenamiento") & (mismo["particion_b"] == "validacion"))]
out(f"   Pares del mismo ojo entre particiones: {(mismo['particion_a'] != mismo['particion_b']).sum()}"
    f" (entrenamiento-validación: {len(fuga_val)}, con prueba: {len(exc)})")
out(f"   Imágenes de prueba excluidas de la evaluación final: {len(exc)} -> {', '.join(exc['codigo'])}")
prueba = part[part["particion"] == "prueba"]
quedan = prueba[~prueba["codigo"].isin(exc["codigo"])]
out(f"   Prueba efectiva: {len(quedan)} imágenes ({quedan['etiqueta'].sum()} glaucoma / {(quedan['etiqueta'] == 0).sum()} sanas)")

# Figura para el informe: pares del mismo ojo que cruzan particiones (arriba y abajo de cada columna)
cruzan = mismo[mismo["particion_a"] != mismo["particion_b"]]
if len(cruzan):
    fig, ejes = plt.subplots(2, len(cruzan), figsize=(2.1 * len(cruzan), 4.4), squeeze=False)
    for k, r in enumerate(cruzan.itertuples()):
        for fila, cod, p, y in [(0, r.codigo_a, r.particion_a, r.etiqueta_a), (1, r.codigo_b, r.particion_b, r.etiqueta_b)]:
            ax = ejes[fila][k]
            with Image.open(data_dir / archivo_de[cod]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{cod} ({p[:5]}.) y={y}", fontsize=7)
            ax.axis("off")
        ejes[1][k].text(0.5, -0.08, f"d = {r.distancia_phash}", transform=ejes[1][k].transAxes,
                        ha="center", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "pares_mismo_ojo.png", dpi=150)
    out("   Figura para el informe: figuras/pares_mismo_ojo.png")

(RESULTADOS / "verificacion_particion.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
