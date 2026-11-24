"""Comparación base / V1 / V2 / V3 sobre las MISMAS 91 imágenes de validación y elección de la versión final.

Regla fijada en config.yaml antes de comparar: mayor AUC de validación; si otra versión más simple
queda a menos de `empate_auc`, gana la más simple. Umbral de la elegida: el menor que mantiene en
validación una especificidad >= a la de la línea base (0,632). Con esto se congelan versión,
preprocesamiento y umbral ANTES de abrir la prueba.
Salidas: resultados/comparacion_validacion.csv, resultados/predicciones_validacion_todas.csv,
         resultados/analisis_comparacion.txt, resultados/version_elegida.json,
         resultados/particion_final.csv
"""
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

from comun import (MODELOS, RESULTADOS, bootstrap_pareado, cargar_config, guardar_json, leer_json, metricas,
                   resumen, tabla_por_caso, umbral_por_especificidad)
from modelos import DESCRIPCION

cfg = cargar_config()
seed, nb, sel = cfg["seed"], cfg["n_bootstrap"], cfg["seleccion"]
VERSIONES = sel["orden_simplicidad"]

# Puntuaciones de validación de cada versión, alineadas por código
base = pd.read_csv(RESULTADOS / "predicciones_validacion.csv")
punt = pd.DataFrame({"codigo": base["codigo"], "etiqueta": base["etiqueta"], "base": base["prob_glaucoma"]})
for v in VERSIONES[1:]:
    p = pd.read_csv(RESULTADOS / f"predicciones_validacion_{v}.csv")
    punt = punt.merge(p[["codigo", "puntuacion"]].rename(columns={"puntuacion": v}), on="codigo", validate="1:1")
assert len(punt) == 91
y = punt["etiqueta"].to_numpy()

auc_ent = {"base": leer_json(RESULTADOS / "metricas_validacion.json")["AUC_entrenamiento"]}
for v in VERSIONES[1:]:
    auc_ent[v] = leer_json(RESULTADOS / f"metricas_validacion_{v}.json")["AUC_entrenamiento"]

filas, res = [], {}
for v in VERSIONES:
    s = punt[v].to_numpy()
    r = resumen(y, s, cfg["umbral"], nb, seed)
    u_esp = umbral_por_especificidad(y, s, sel["especificidad_minima"])
    m_esp, (vp2, fn2, fp2, vn2) = metricas(y, (s >= u_esp).astype(int), s)
    d = bootstrap_pareado(y, punt["base"].to_numpy(), s, nb, seed) if v != "base" else None
    res[v] = r
    M = r["metricas"]
    filas.append({
        "version": v, "descripcion": DESCRIPCION[v], "particion": "validacion", "n": r["n"],
        "VP": r["VP"], "FN": r["FN"], "FP": r["FP"], "VN": r["VN"],
        "sensibilidad": round(M["sensibilidad"]["valor"], 3), "sens_den": r["denominadores"]["sensibilidad"],
        "especificidad": round(M["especificidad"]["valor"], 3), "esp_den": r["denominadores"]["especificidad"],
        "AUC": round(M["AUC"]["valor"], 3),
        "AUC_IC95": f"{M['AUC']['IC95'][0]:.2f}-{M['AUC']['IC95'][1]:.2f}",
        "F1": round(M["F1"]["valor"], 3),
        "AUC_entrenamiento": round(auc_ent[v], 3),
        "dAUC_vs_base": round(d["valor"], 3) if d else 0.0,
        "dAUC_IC95": f"{d['IC95'][0]:.2f}-{d['IC95'][1]:.2f}" if d else "",
        "umbral_esp_base": round(u_esp, 4),
        "sens_con_esp_base": f"{vp2}/{vp2 + fn2}", "esp_con_esp_base": f"{vn2}/{vn2 + fp2}",
    })
comp = pd.DataFrame(filas)
comp.to_csv(RESULTADOS / "comparacion_validacion.csv", index=False)

# Tabla por caso con las cuatro versiones (umbral 0,5)
pd.concat([tabla_por_caso(punt["codigo"], "validacion", v, y, punt[v], cfg["umbral"]) for v in VERSIONES]
          ).to_csv(RESULTADOS / "predicciones_validacion_todas.csv", index=False)

# --- Elección según la regla fijada
aucs = comp.set_index("version")["AUC"]
mejor = aucs.idxmax()
candidatas = [v for v in VERSIONES if aucs[v] >= aucs[mejor] - sel["empate_auc"]]
elegida = candidatas[0]
s_eleg = punt[elegida].to_numpy()
umbral = umbral_por_especificidad(y, s_eleg, sel["especificidad_minima"])
r_eleg = resumen(y, s_eleg, umbral, nb, seed)

archivos = {"base": ["modelo_base.joblib"], "v1": ["modelo_v1.joblib"], "v2": ["modelo_v2.joblib"],
            "v3": ["modelo_v3.pt"]}[elegida]
hashes = {a: hashlib.sha256((MODELOS / a).read_bytes()).hexdigest() for a in archivos}
guardar_json({
    "version": elegida, "descripcion": DESCRIPCION[elegida],
    "regla": f"mayor AUC de validación; empate si diferencia < {sel['empate_auc']} -> la más simple "
             f"(orden {VERSIONES})",
    "AUC_validacion": {v: float(aucs[v]) for v in VERSIONES},
    "umbral": umbral,
    "regla_umbral": f"menor umbral con especificidad >= {sel['especificidad_minima']} en validación",
    "validacion_con_umbral": {k: r_eleg[k] for k in ["VP", "FN", "FP", "VN", "denominadores"]},
    "preprocesamiento": ("RGB -> 256x256 bilineal -> 8 rasgos de color/palidez" if elegida == "base" else
                         f"RGB -> {cfg['cnn']['img_size']}x{cfg['cnn']['img_size']} bilineal -> "
                         "normalización ImageNet -> ResNet-18 (" + cfg["cnn"]["pesos"] + ")"),
    "archivos_modelo": archivos, "sha256": hashes,
    "fecha_congelado": datetime.now().isoformat(timespec="seconds"),
}, RESULTADOS / "version_elegida.json")

# --- Partición final con exclusiones (inventario + asignación + inclusión en la evaluación)
part = pd.read_csv(RESULTADOS / "particion.csv")
excl = pd.read_csv(RESULTADOS / "prueba_excluidos.csv")
motivo = {c: f"mismo ojo que {g} ({pg}), pHash {d}" for c, g, pg, d in
          zip(excl["codigo"], excl["gemelo"], excl["particion_gemelo"], excl["distancia_phash"])}
part["incluido_en_evaluacion"] = ~part["codigo"].isin(motivo)
part["motivo_exclusion"] = part["codigo"].map(motivo).fillna("")
part.to_csv(RESULTADOS / "particion_final.csv", index=False)

# --- Análisis complementario
lineas = []
def out(t=""):
    print(t); lineas.append(t)

out(f"Comparación en validación (n = {len(y)}: {int(y.sum())} glaucoma / {int((y == 0).sum())} sanos), umbral 0,5")
out(comp[["version", "VP", "FN", "FP", "VN", "sens_den", "esp_den", "AUC", "AUC_IC95", "AUC_entrenamiento",
          "dAUC_vs_base", "dAUC_IC95", "sens_con_esp_base", "esp_con_esp_base"]].to_string(index=False))
out(f"\nElegida: {elegida} ({DESCRIPCION[elegida]}); candidatas por empate: {candidatas}")
out(f"Umbral congelado: {umbral:.4f} -> validación sens {r_eleg['denominadores']['sensibilidad']}, "
    f"esp {r_eleg['denominadores']['especificidad']}")

car = pd.read_csv(RESULTADOS / "caracteristicas.csv").set_index("codigo")
brillo = (0.299 * car["media_R"] + 0.587 * car["media_G"] + 0.114 * car["media_B"]).loc[punt["codigo"]].to_numpy()
tercil = pd.qcut(brillo, 3, labels=["oscura", "media", "clara"])
out("\nTasa de error (umbral 0,5) por tercil de brillo del disco en validación:")
for v in VERSIONES:
    err = ((punt[v].to_numpy() >= cfg["umbral"]).astype(int) != y)
    out(f"  {v:5s} " + "  ".join(f"{t}: {err[tercil == t].sum()}/{(tercil == t).sum()}"
                                for t in ["oscura", "media", "clara"]))

for caso in ["train/233", "train/027"]:
    fila = punt[punt["codigo"] == caso].iloc[0]
    out(f"\n{caso} (etiqueta {int(fila['etiqueta'])}): " +
        ", ".join(f"{v} p={fila[v]:.3f}" for v in VERSIONES))
sin = punt[punt["codigo"] != "train/027"]
out("AUC sin train/027 (gemelo en entrenamiento): " +
    ", ".join(f"{v} {metricas(sin['etiqueta'].to_numpy(), (sin[v] >= 0.5).astype(int), sin[v].to_numpy())[0]['AUC']:.3f}"
              for v in VERSIONES))
(RESULTADOS / "analisis_comparacion.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
