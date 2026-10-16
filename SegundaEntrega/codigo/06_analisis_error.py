"""Cifras del análisis de resultados y del error de la Sección 4 del informe.

Usa solo resultados/caracteristicas.csv y resultados/predicciones_validacion.csv
(no vuelve a abrir imágenes ni toca la prueba). Reentrena el mismo modelo de
04_linea_base.py con el entrenamiento para descomponer el logit del caso analizado.

Salida: resultados/analisis_error.txt
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comun import RESULTADOS, cargar_config

cfg = cargar_config()
CASO = "train/233"          # falso negativo analizado
GEMELO_VAL = "train/027"    # caso de validación con gemelo en entrenamiento
F = ["media_R", "media_G", "media_B", "std_R", "std_G", "std_B", "palidez_G", "palidez_R"]

car = pd.read_csv(RESULTADOS / "caracteristicas.csv")
pred = pd.read_csv(RESULTADOS / "predicciones_validacion.csv")
ent = car[car["particion"] == "entrenamiento"]

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(class_weight="balanced", C=cfg["C"], max_iter=1000)),
]).fit(ent[F], ent["etiqueta"])
sc, lr = modelo.named_steps["scaler"], modelo.named_steps["lr"]

lineas = []
def out(t=""):
    print(t); lineas.append(t)

# 1. Distribución de probabilidades en validación
out("1. Probabilidades en validación")
out(f"   Mediana por clase: {pred.groupby('etiqueta')['prob_glaucoma'].median().round(3).to_dict()}")
fp, fn = pred[pred["tipo"] == "FP"], pred[pred["tipo"] == "FN"]
out(f"   FP con p < 0,60: {(fp['prob_glaucoma'] < 0.60).sum()} de {len(fp)}")
out(f"   FN con p entre 0,39 y 0,49: {fn['prob_glaucoma'].between(0.39, 0.49).sum()} de {len(fn)}")

# 2. Características en entrenamiento
out("\n2. Características en entrenamiento")
out(f"   media_R > 0,95: {(ent['media_R'] > 0.95).mean() * 100:.1f} % | min {ent['media_R'].min():.3f} "
    f"| media {ent['media_R'].mean():.4f} | desv {ent['media_R'].std():.4f}")
out(f"   Mediana de palidez_G por clase: {ent.groupby('etiqueta')['palidez_G'].median().round(3).to_dict()}")

# 3. Caso analizado
x = car.loc[car["codigo"] == CASO, F].iloc[0].astype(float)
z = (x.to_numpy() - sc.mean_) / sc.scale_
contrib = z * lr.coef_[0]
p_orig = modelo.predict_proba(pd.DataFrame([x.to_numpy()], columns=F))[0, 1]
out(f"\n3. Caso {CASO}")
out(f"   p = {p_orig:.3f} | media_R = {x['media_R']:.3f} | media_G = {x['media_G']:.3f} "
    f"(mín. de media_G en entrenamiento: {ent['media_G'].min():.3f})")
out(f"   media_R a {(x['media_R'] - ent['media_R'].mean()) / ent['media_R'].std():.1f} desviaciones de la media de entrenamiento")
out("   Aporte de cada característica al logit: "
    + ", ".join(f"{k} {v:+.1f}" for k, v in zip(F, contrib)) + f", intercepto {lr.intercept_[0]:+.2f}")
x_rec = np.clip(x.to_numpy(), ent[F].min().to_numpy(), ent[F].max().to_numpy())
p_rec = modelo.predict_proba(pd.DataFrame([x_rec], columns=F))[0, 1]
out(f"   Prueba (1): características recortadas al rango de entrenamiento -> p = {p_rec:.3f}")

# 4. Efecto del gemelo entrenamiento-validación
sin = pred[pred["codigo"] != GEMELO_VAL]
vn = ((sin["etiqueta"] == 0) & (sin["prediccion"] == 0)).sum()
out(f"\n4. Validación sin {GEMELO_VAL} ({pred.loc[pred['codigo'] == GEMELO_VAL, 'tipo'].iloc[0]}, "
    f"p = {pred.loc[pred['codigo'] == GEMELO_VAL, 'prob_glaucoma'].iloc[0]:.3f})")
out(f"   Especificidad {vn}/{(sin['etiqueta'] == 0).sum()} = {vn / (sin['etiqueta'] == 0).sum():.3f} "
    f"| AUC {roc_auc_score(sin['etiqueta'], sin['prob_glaucoma']):.3f}")

(RESULTADOS / "analisis_error.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
