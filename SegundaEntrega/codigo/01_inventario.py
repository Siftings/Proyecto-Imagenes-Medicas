"""Inventario de la copia de ORIGA-light: cruza archivos con CSV, abre cada imagen y calcula MD5.

Salida: resultados/inventario.csv
"""
import hashlib

import pandas as pd
from PIL import Image

from comun import RESULTADOS, cargar_config, codigo_de

cfg = cargar_config()
data_dir = cfg["data_dir"]

# Etiquetas de los dos CSV
etiquetas = {}
nombres_csv = []
for part in ["train", "test"]:
    df = pd.read_csv(data_dir / f"ORIGA_{part}.csv")
    for nombre, y in zip(df["ImageName"], df["glaucoma"]):
        nombres_csv.append(nombre)
        etiquetas[nombre] = int(y)

# Archivos presentes en disco
archivos_disco = sorted(
    str(p.relative_to(data_dir)) for p in (data_dir / "ORIGA").glob("*/*.jpg")
)
todos = sorted(set(nombres_csv) | set(archivos_disco))

filas = []
for nombre in todos:
    ruta = data_dir / nombre
    fila = {
        "codigo": codigo_de(nombre),
        "archivo": nombre,
        "particion_original": ruta.parent.name,
        "etiqueta": etiquetas.get(nombre),
        "ancho": None, "alto": None, "modo": None, "md5": None,
        "legible": False,
    }
    if ruta.exists():
        fila["md5"] = hashlib.md5(ruta.read_bytes()).hexdigest()
        try:
            with Image.open(ruta) as im:
                im.load()
                fila["ancho"], fila["alto"] = im.size
                fila["modo"] = im.mode
            fila["legible"] = True
        except Exception:
            pass
    filas.append(fila)

inv = pd.DataFrame(filas)
inv["etiqueta"] = inv["etiqueta"].astype("Int64")
inv.to_csv(RESULTADOS / "inventario.csv", index=False)

# Resumen en pantalla
sin_archivo = sorted(set(nombres_csv) - set(archivos_disco))
sin_etiqueta = sorted(set(archivos_disco) - set(nombres_csv))
repetidos_csv = len(nombres_csv) - len(set(nombres_csv))
md5_dup = inv["md5"].dropna().duplicated(keep=False).sum()
leg = inv[inv["legible"]]
print(f"Imágenes en disco: {len(archivos_disco)} | filas en CSV: {len(nombres_csv)}")
print(f"En CSV sin archivo: {len(sin_archivo)} | archivos sin etiqueta: {len(sin_etiqueta)}")
print(f"Ilegibles: {(~inv['legible']).sum()} | nombres repetidos en CSV: {repetidos_csv} | archivos con MD5 repetido: {md5_dup}")
print(f"Modos de color: {leg['modo'].value_counts().to_dict()}")
print(f"Cuadradas: {(leg['ancho'] == leg['alto']).sum()} de {len(leg)}")
print(f"Tamaños distintos: {leg[['ancho', 'alto']].drop_duplicates().shape[0]} "
      f"| mín {leg['ancho'].min()}x{leg['alto'].min()} | máx {leg['ancho'].max()}x{leg['alto'].max()}")
print("\nConteos por partición original y etiqueta:")
print(pd.crosstab(inv["particion_original"], inv["etiqueta"], margins=True, margins_name="total"))
