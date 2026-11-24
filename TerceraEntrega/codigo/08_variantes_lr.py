"""Variantes V1 y V2: rasgos de ResNet-18 congelada + el mismo clasificador de la línea base.

V1: StandardScaler + LR balanceada con C=1 (idéntico a la base salvo la representación).
V2: igual, con C elegido por validación cruzada estratificada SOLO dentro de entrenamiento (AUC).
Ajuste con las 363 de entrenamiento; una evaluación sobre las 91 de validación. La prueba no se toca.
Salidas: resultados/predicciones_validacion_{v1,v2}.csv, resultados/metricas_validacion_{v1,v2}.json,
         modelos/modelo_{v1,v2}.joblib
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comun import MODELOS, RESULTADOS, cargar_config, fijar_semillas, guardar_json, resumen, tabla_por_caso
from modelos import DESCRIPCION

cfg = cargar_config()
seed = cfg["seed"]
fijar_semillas(seed)

d = np.load(RESULTADOS / "caracteristicas_cnn.npz")  # solo arreglos numéricos y de texto, sin pickle
ent, val = d["particion"] == "entrenamiento", d["particion"] == "validacion"
X_ent, y_ent = d["X"][ent], d["etiqueta"][ent].astype(int)
X_val, y_val, cod_val = d["X"][val], d["etiqueta"][val].astype(int), d["codigo"][val]
assert len(y_ent) == 363 and len(y_val) == 91


def pipeline(C):
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", C=C, max_iter=5000))])


v1 = pipeline(cfg["C"]).fit(X_ent, y_ent)

busqueda = GridSearchCV(pipeline(1.0), {"lr__C": cfg["lr_cv"]["Cs"]}, scoring="roc_auc",
                        cv=StratifiedKFold(cfg["lr_cv"]["folds"], shuffle=True, random_state=seed))
busqueda.fit(X_ent, y_ent)
v2 = busqueda.best_estimator_
cv_tabla = {str(c): float(m) for c, m in zip(cfg["lr_cv"]["Cs"], busqueda.cv_results_["mean_test_score"])}

for nombre, modelo, extra in [("v1", v1, {"C": cfg["C"]}),
                              ("v2", v2, {"C": busqueda.best_params_["lr__C"],
                                          "AUC_cv_entrenamiento_por_C": cv_tabla})]:
    joblib.dump(modelo, MODELOS / f"modelo_{nombre}.joblib")
    prob = modelo.predict_proba(X_val)[:, 1]
    auc_ent = roc_auc_score(y_ent, modelo.predict_proba(X_ent)[:, 1])
    tabla_por_caso(cod_val, "validacion", nombre, y_val, prob, cfg["umbral"]).to_csv(
        RESULTADOS / f"predicciones_validacion_{nombre}.csv", index=False)
    salida = {"version": nombre, "descripcion": DESCRIPCION[nombre],
              **resumen(y_val, prob, cfg["umbral"], cfg["n_bootstrap"], seed),
              "AUC_entrenamiento": auc_ent, **extra}
    guardar_json(salida, RESULTADOS / f"metricas_validacion_{nombre}.json")
    m = salida["metricas"]
    print(f"{nombre}: C={extra['C']} | AUC val {m['AUC']['valor']:.3f} "
          f"[{m['AUC']['IC95'][0]:.2f}, {m['AUC']['IC95'][1]:.2f}] | AUC ent {auc_ent:.3f} | "
          f"sens {salida['denominadores']['sensibilidad']} esp {salida['denominadores']['especificidad']}")
print("AUC de CV (entrenamiento) por C:", {k: round(v, 3) for k, v in cv_tabla.items()})
