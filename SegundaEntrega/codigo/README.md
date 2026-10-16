# Entrega 2 — Proyecto B: glaucoma en ORIGA-light

Código de la segunda entrega de Imágenes Médicas (34740, PUJ): inventario de los datos, partición
entrenamiento/validación/prueba con su verificación, y una línea base simple (8 características de
color y palidez + regresión logística balanceada) evaluada una sola vez sobre la validación.
**El conjunto de prueba no se usa en esta entrega.**

## 1. Instalación

Requiere Python 3.10 o superior. No usa redes neuronales ni GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Datos

El conjunto **no** se incluye en este repositorio ni en el ZIP. Descárgalo de Kaggle:
<https://www.kaggle.com/datasets/ferencjuhsz/origa-retinal-fundus-image-dataset>

Descomprime y deja la carpeta `archive/` con esta estructura:

```
archive/
├── ORIGA/train/*.jpg   (454)
├── ORIGA/test/*.jpg    (196)
├── ORIGA_train.csv
└── ORIGA_test.csv
```

Ajusta `data_dir` en `config.yaml` (por defecto `../archive`, es decir, al lado de esta carpeta).

## 3. Ejecución

Desde esta carpeta, en orden:

```bash
python 01_inventario.py           # resultados/inventario.csv
python 02_particion.py            # resultados/particion.csv
python 03_verificar_particion.py  # resultados/verificacion_particion.txt, figuras/pares_phash.png
python 04_linea_base.py           # caracteristicas.csv, predicciones_validacion.csv, metricas_validacion.json
python 05_figuras.py              # figuras/casos_validacion.png, matriz_confusion.png, curva_roc.png
```

Todo corre en CPU en menos de un minuto (el paso más lento es el pHash de las 650 imágenes).

## 4. Parámetros (`config.yaml`)

| Clave | Valor | Uso |
|---|---|---|
| `seed` | 42 | partición, bootstrap |
| `val_size` | 0.2 | fracción de `ORIGA/train` para validación (estratificada) |
| `img_size` | 256 | lado de la imagen redimensionada |
| `r_centro`, `r_disco` | 38, 102 | radios (px) de las regiones centro y disco |
| `C`, `umbral` | 1.0, 0.5 | regresión logística y umbral de decisión (fijos, sin ajuste) |
| `n_bootstrap` | 1000 | remuestreos para los IC 95 % |
| `phash_threshold` | 8 | distancia de Hamming máxima para marcar casi duplicados |

## 5. Contenido

- `comun.py`: lectura de `config.yaml` y rutas compartidas.
- `resultados/`: tablas y métricas generadas.
- `figuras/`: figuras generadas.
