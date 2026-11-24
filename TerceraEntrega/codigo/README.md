# Entrega final: Proyecto B, glaucoma en ORIGA-light (Equipo 02)

Este es el código de la entrega final de Imágenes Médicas (34740, PUJ). Hace cuatro cosas:
1. Inventaría el conjunto de imágenes y arma la partición entrenamiento/validación/prueba. Luego comprueba que no haya imágenes compartidas entre particiones.
2. Compara cuatro versiones del clasificador sobre las mismas 91 imágenes de validación: la línea base y tres variantes.
3. Congela la versión elegida.
4. Evalúa esa versión **una sola vez** sobre la prueba reservada.

La clase positiva es glaucoma (1).

## 1. Software e instalación

Se probó con Python 3.13.13, numpy 2.3, pandas 2.3, scikit-learn 1.9.1, Pillow 12, torch 2.15 y torchvision 0.29, en macOS (Apple M4) usando solo la CPU.
Las versiones exactas de cada ejecución quedan en los JSON de `resultados/`.

```bash
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

La primera vez que se usa ResNet-18, torchvision descarga los pesos públicos de ImageNet (`resnet18-f37072fd.pth`, 45 MB) desde `download.pytorch.org` y los guarda en `~/.cache/torch`. Para eso se necesita internet.

## 2. Datos (no se incluyen)

Hay que descargar el conjunto desde Kaggle: <https://www.kaggle.com/datasets/ferencjuhsz/origa-retinal-fundus-image-dataset> (licencia MIT).
Luego se ubica la carpeta `archive/` así:

```
archive/
├── ORIGA/train/*.jpg   (454)
├── ORIGA/test/*.jpg    (196)
├── ORIGA_train.csv
└── ORIGA_test.csv
```

La ruta de los datos se configura en `data_dir`, dentro de `config.yaml`. Por defecto es `../archive`, es decir, al lado de la carpeta `codigo/`. Las imágenes se identifican con códigos seudónimos de la forma `train/093` o `test/240`.

## 3. Ejecución

La semilla es 42 para todo (partición, bootstrap y torch). Todos los comandos se corren desde esta carpeta.

**A. Datos, partición y línea base (Entrega 2, < 1 min)**
```bash
python 01_inventario.py           # resultados/inventario.csv
python 02_particion.py            # resultados/particion.csv (363 / 91 / 196)
python 03_verificar_particion.py  # verificacion_particion.txt, revision_pares.csv, prueba_excluidos.csv
python 04_linea_base.py           # base: caracteristicas.csv, predicciones_validacion.csv, metricas_validacion.json, modelos/modelo_base.joblib
python 05_figuras.py              # figuras de la base en validación (Entrega 2)
python 06_analisis_error.py       # análisis del FN train/233 de la base
```

**B. Entrenamiento de las variantes (≈ 6 min en CPU; se puede omitir porque los pesos van en `modelos/`)**
```bash
python 07_extraer_cnn.py          # rasgos ResNet-18 congelada (650 x 512)
python 08_variantes_lr.py         # V1 y V2 -> modelos/modelo_v1.joblib, modelo_v2.joblib
python 09_ajuste_fino.py          # V3 (ajuste fino, 20 épocas) -> modelos/modelo_v3.pt
```

**C. Comparación, evaluación e inferencia (usa los pesos guardados, < 1 min)**
```bash
python 10_comparacion_validacion.py   # tabla base/V1/V2/V3, version_elegida.json (versión + umbral congelados)
python 11_evaluacion_prueba.py        # SOLO la versión elegida sobre la prueba (192); deja un candado
python 12_figuras_final.py            # figuras de validación y prueba (incluye Grad-CAM)
python predecir.py ../archive/ORIGA/test/240.jpg   # inferencia de una imagen con la versión congelada
```

`11_evaluacion_prueba.py` escribe `resultados/prueba_ejecutada.lock` y no vuelve a correr sin `--forzar`. Esto deja constancia de que la prueba se evaluó una sola vez, después de fijar la versión.
Si se quiere reproducir el resultado sobre una copia nueva del ZIP, se usa `python 11_evaluacion_prueba.py --forzar`. Las cifras deben coincidir con `metricas_prueba.json`.

**Variación numérica:** en CPU, con la semilla fija, los resultados se repiten exactamente. En GPU o MPS (`cnn.dispositivo` en `config.yaml`), el ajuste fino puede dar diferencias pequeñas en las puntuaciones.

## 4. Versiones comparadas (`config.yaml`)

| Versión | Representación | Clasificador | Qué cambia frente a la base |
|---|---|---|---|
| base | 8 rasgos de color y palidez en el disco (256 px) | StandardScaler + LR balanceada, C=1 | — |
| V1 | 512 rasgos de ResNet-18 congelada (ImageNet, 224 px) | igual que la base (C=1) | solo la representación |
| V2 | igual que V1 | LR balanceada, C elegido por CV de 5 pliegues **dentro de entrenamiento** | representación + regularización |
| V3 | ResNet-18 con `layer4` y la capa final ajustadas | salida sigmoide, BCE ponderada, AdamW 1e-4, 20 épocas fijas, aumento | red ajustada a los datos |

Regla de selección, fijada antes de comparar: gana la de mayor AUC en validación. Si la diferencia es menor de 0,01, gana la más simple.
El umbral de la elegida es el menor que mantiene en validación una especificidad ≥ 0,632, que es la de la base con umbral 0,5.

## 5. Salidas principales

| Archivo | Contenido |
|---|---|
| `resultados/particion_final.csv` | inventario y asignación por imagen, más `incluido_en_evaluacion` y el motivo de exclusión (4 de prueba) |
| `resultados/verificacion_particion.txt` | comprobación de códigos, MD5 y pHash entre particiones |
| `resultados/comparacion_validacion.csv` | base, V1, V2 y V3 sobre las mismas 91: matriz, sensibilidad y especificidad con denominadores, AUC con IC 95 %, ΔAUC frente a la base |
| `resultados/predicciones_validacion_todas.csv` | tabla por caso de validación para las 4 versiones |
| `resultados/version_elegida.json` | versión, umbral, preprocesamiento y SHA-256 de los pesos congelados |
| `resultados/predicciones_prueba.csv` | tabla por caso de prueba (ID, etiqueta, puntuación, predicción, tipo, incluido) |
| `resultados/metricas_prueba.json` | métricas de prueba (192 imágenes; también 196 y 191 como análisis de sensibilidad) |
| `resultados/analisis_comparacion.txt` | errores por exposición, `train/233` y efecto de `train/027` |
| `modelos/` | `modelo_base.joblib`, `modelo_v1.joblib`, `modelo_v2.joblib` y `modelo_v3.pt` (pesos entrenados) |
| `figuras/` | figuras del informe; en cada una se indica método, partición y n |

Las figuras muestran imágenes del conjunto, que la copia de Kaggle distribuye con licencia MIT. El ZIP no incluye el conjunto completo.
