"""Rasgos de ResNet-18 congelada (ImageNet) para las 650 imágenes: vector de 512 de la capa avgpool.

La red no se ajusta con nuestros datos, así que extraer rasgos de prueba no usa sus etiquetas.
Aun así, los rasgos de prueba no se leen hasta 11_evaluacion_prueba.py.
Salida: resultados/caracteristicas_cnn.npz (codigo, particion, etiqueta, X de 650 x 512)
"""
import time

import numpy as np
import pandas as pd

from comun import RESULTADOS, cargar_config, fijar_semillas
from modelos import rasgos_cnn

cfg = cargar_config()
fijar_semillas(cfg["seed"])

part = pd.read_csv(RESULTADOS / "particion.csv")
t0 = time.time()
X = rasgos_cnn([cfg["data_dir"] / f for f in part["archivo"]], cfg)
np.savez_compressed(RESULTADOS / "caracteristicas_cnn.npz", codigo=part["codigo"].to_numpy().astype(str),
                    particion=part["particion"].to_numpy().astype(str), etiqueta=part["etiqueta"].to_numpy(), X=X)
print(f"Rasgos ResNet-18: {X.shape} en {time.time() - t0:.1f} s "
      f"({cfg['cnn']['pesos']}, {cfg['cnn']['img_size']} px, {cfg['cnn']['dispositivo']})")
