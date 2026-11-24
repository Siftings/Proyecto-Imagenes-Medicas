"""Evaluación ÚNICA de la versión elegida sobre la prueba reservada.

Lee resultados/version_elegida.json (versión, pesos y umbral congelados en 10_comparacion_validacion.py)
y la aplica sin cambios a las 196 imágenes de prueba. Las métricas principales se calculan sobre las
192 incluidas (sin las 4 con un gemelo en entrenamiento); 196 y "sin test/175" son análisis de sensibilidad.

Candado: la primera ejecución escribe resultados/prueba_ejecutada.lock con la fecha y el hash de la
versión elegida. Una segunda ejecución se detiene, salvo con --forzar (que queda registrado).
Salidas: resultados/predicciones_prueba.csv, resultados/metricas_prueba.json, resultados/prueba_ejecutada.lock
"""
import argparse
import hashlib
import sys
from datetime import datetime

import pandas as pd

from comun import MODELOS, RESULTADOS, cargar_config, fijar_semillas, guardar_json, leer_json, resumen, tabla_por_caso
from modelos import Version

ap = argparse.ArgumentParser()
ap.add_argument("--forzar", action="store_true", help="repetir la evaluación (queda registrado en el candado)")
args = ap.parse_args()

cfg = cargar_config()
fijar_semillas(cfg["seed"])
candado = RESULTADOS / "prueba_ejecutada.lock"
eleg_txt = (RESULTADOS / "version_elegida.json").read_bytes()
huella = hashlib.sha256(eleg_txt).hexdigest()

if candado.exists() and not args.forzar:
    sys.exit(f"La prueba ya se evaluó (ver {candado.name}):\n{candado.read_text()}\n"
             "No se repite para no ajustar según la prueba. Use --forzar solo para reproducir, no para decidir.")

eleg = leer_json(RESULTADOS / "version_elegida.json")
for a, h in eleg["sha256"].items():  # los pesos deben ser exactamente los congelados
    assert hashlib.sha256((MODELOS / a).read_bytes()).hexdigest() == h, f"{a} cambió después de congelar"

part = pd.read_csv(RESULTADOS / "particion_final.csv")
prueba = part[part["particion"] == "prueba"].reset_index(drop=True)
version = Version(eleg["version"], cfg)
s = version.puntuar([cfg["data_dir"] / f for f in prueba["archivo"]])
umbral = eleg["umbral"]

tabla = tabla_por_caso(prueba["codigo"], "prueba", eleg["version"], prueba["etiqueta"], s, umbral)
tabla["incluido_en_evaluacion"] = prueba["incluido_en_evaluacion"].to_numpy()
tabla.to_csv(RESULTADOS / "predicciones_prueba.csv", index=False)

inc = tabla[tabla["incluido_en_evaluacion"]]
sin175 = inc[inc["codigo"] != "test/175"]
nb, seed = cfg["n_bootstrap"], cfg["seed"]
salida = {
    "version": eleg["version"], "descripcion": eleg["descripcion"], "particion": "prueba",
    "umbral_congelado": umbral, "clase_positiva": "glaucoma (1)",
    "principal_192": resumen(inc["etiqueta"], inc["puntuacion"], umbral, nb, seed),
    "sensibilidad_196_sin_excluir": resumen(tabla["etiqueta"], tabla["puntuacion"], umbral, nb, seed),
    "sensibilidad_191_sin_test175": resumen(sin175["etiqueta"], sin175["puntuacion"], umbral, nb, seed),
    "huella_version_elegida": huella,
}
guardar_json(salida, RESULTADOS / "metricas_prueba.json")

registro = (f"fecha: {datetime.now().isoformat(timespec='seconds')}\nversion: {eleg['version']}\n"
            f"umbral: {umbral}\nsha256 version_elegida.json: {huella}\n")
if candado.exists():
    registro = candado.read_text() + "\n[REPETIDA CON --forzar]\n" + registro
candado.write_text(registro, encoding="utf-8")

p = salida["principal_192"]
M = p["metricas"]
print(f"Prueba ({eleg['version']}, umbral {umbral:.4f}) n={p['n']} ({p['n_glaucoma']} glaucoma / {p['n_sano']} sanos)")
print(f"  VP={p['VP']} FN={p['FN']} FP={p['FP']} VN={p['VN']}")
for k in ["sensibilidad", "especificidad", "AUC", "F1"]:
    print(f"  {k:13s} {M[k]['valor']:.3f}  IC95 [{M[k]['IC95'][0]:.3f}, {M[k]['IC95'][1]:.3f}]")
print(f"  denominadores: sens {p['denominadores']['sensibilidad']}, esp {p['denominadores']['especificidad']}")
