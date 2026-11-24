"""Inferencia con la versión congelada sobre una o varias imágenes (demostración de la sustentación).

Uso:
    python predecir.py ../archive/ORIGA/test/240.jpg [otra.jpg ...]
    python predecir.py --version base ../archive/ORIGA/test/240.jpg   # otra versión, umbral 0,5

Imprime puntuación de glaucoma, umbral y clase. Si la imagen está en la partición, muestra también
su etiqueta de referencia. No reentrena nada: usa modelos/ y resultados/version_elegida.json.
"""
import argparse
from pathlib import Path

import pandas as pd

from comun import NOMBRE_CLASE, RESULTADOS, cargar_config, codigo_de, fijar_semillas, leer_json
from modelos import Version

ap = argparse.ArgumentParser()
ap.add_argument("imagenes", nargs="+")
ap.add_argument("--version", choices=["base", "v1", "v2", "v3"], help="por defecto, la versión elegida")
args = ap.parse_args()

cfg = cargar_config()
fijar_semillas(cfg["seed"])
eleg = leer_json(RESULTADOS / "version_elegida.json")
nombre = args.version or eleg["version"]
umbral = eleg["umbral"] if nombre == eleg["version"] else cfg["umbral"]
version = Version(nombre, cfg)

part = pd.read_csv(RESULTADOS / "particion.csv").set_index("codigo")
s = version.puntuar([Path(r) for r in args.imagenes])
print(f"{version.descripcion} | umbral {umbral:.4f}")
for ruta, p in zip(args.imagenes, s):
    cod = codigo_de(ruta)
    ref = (f" | referencia: {NOMBRE_CLASE[int(part.loc[cod, 'etiqueta'])]} ({part.loc[cod, 'particion']})"
           if cod in part.index else "")
    print(f"{cod}: puntuación {p:.4f} -> {NOMBRE_CLASE[int(p >= umbral)]}{ref}")
