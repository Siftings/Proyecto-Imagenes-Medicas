"""Análisis exploratorio DESPUÉS de la evaluación única en prueba (no cambia versión ni umbral).

Cifras de la Sección 5 del informe: distribución de puntuaciones de V2 en validación y prueba,
qué habría pasado con el umbral 0,5 (solo para explicar la caída de especificidad), errores por
brillo y datos de los dos casos analizados. Lee resultados/ y abre imágenes solo para medir brillo.
Salida: resultados/analisis_prueba.txt
"""
import pandas as pd

from comun import RESULTADOS, cargar_config, extraer_color, leer_json, metricas

cfg = cargar_config()
eleg = leer_json(RESULTADOS / "version_elegida.json")
pp = pd.read_csv(RESULTADOS / "predicciones_prueba.csv")
inc = pp[pp["incluido_en_evaluacion"]].copy()
val = pd.read_csv(RESULTADOS / f"predicciones_validacion_{eleg['version']}.csv")
part = pd.read_csv(RESULTADOS / "particion.csv").set_index("codigo")
car = pd.read_csv(RESULTADOS / "caracteristicas.csv")
casos = pd.read_csv(RESULTADOS / "casos_analizados_prueba.csv")

lineas = []
def out(t=""):
    print(t); lineas.append(t)

def lum(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

out(f"ANÁLISIS EXPLORATORIO POSTERIOR A LA PRUEBA ({eleg['version']}, umbral congelado {eleg['umbral']:.4f})")
out("No se usa para decidir: la versión y el umbral ya estaban fijados.\n")
out("1. Puntuaciones (mediana por clase; 0 = sano, 1 = glaucoma)")
out(f"   Validación (n={len(val)}): {val.groupby('etiqueta')['puntuacion'].median().round(3).to_dict()} "
    f"rango {val['puntuacion'].min():.3f}-{val['puntuacion'].max():.3f}")
out(f"   Prueba (n={len(inc)}):     {inc.groupby('etiqueta')['puntuacion'].median().round(3).to_dict()} "
    f"rango {inc['puntuacion'].min():.3f}-{inc['puntuacion'].max():.3f}")
fp, fn = inc[inc["tipo"] == "FP"], inc[inc["tipo"] == "FN"]
out(f"   FP con p < 0,52: {(fp['puntuacion'] < 0.52).sum()} de {len(fp)}; FN entre "
    f"{fn['puntuacion'].min():.3f} y {fn['puntuacion'].max():.3f}")
m, (vp, fnn, fpp, vn) = metricas(inc["etiqueta"].to_numpy(), (inc["puntuacion"] >= 0.5).astype(int).to_numpy(),
                                 inc["puntuacion"].to_numpy())
out(f"   Con umbral 0,5 (hipotético, no elegido): VP={vp} FN={fnn} FP={fpp} VN={vn} -> sens {vp}/{vp + fnn}, "
    f"esp {vn}/{vn + fpp}")

ent = car[car["particion"] == "entrenamiento"]
L_ent = lum(ent["media_R"], ent["media_G"], ent["media_B"])
inc["brillo"] = [lum(*extraer_color(cfg["data_dir"] / part.loc[c, "archivo"], cfg)[:3]) for c in inc["codigo"]]
t = pd.qcut(inc["brillo"], 3, labels=["oscura", "media", "clara"])
err = inc["prediccion"] != inc["etiqueta"]
out("\n2. Brillo del disco (luminancia media en el círculo de radio 102 px)")
out(f"   Entrenamiento: media {L_ent.mean():.3f}, desv {L_ent.std():.3f}; prueba: media {inc['brillo'].mean():.3f}")
out("   Errores por tercil en prueba: " + ", ".join(f"{g} {err[t == g].sum()}/{(t == g).sum()}"
                                              for g in ["oscura", "media", "clara"]))
out("   FP por tercil: " + ", ".join(f"{g} {((inc['tipo'] == 'FP') & (t == g)).sum()} de "
                                     f"{((inc['etiqueta'] == 0) & (t == g)).sum()} sanos"
                                     for g in ["oscura", "media", "clara"]))

out("\n3. Casos analizados")
for _, c in casos.iterrows():
    b = inc.loc[inc["codigo"] == c["codigo"], "brillo"].iloc[0]
    out(f"   {c['rol']}: {c['codigo']} ref {c['etiqueta']} p={c['puntuacion']:.4f} pred {c['prediccion']} ({c['tipo']}), "
        f"distancia al umbral {c['puntuacion'] - eleg['umbral']:+.4f}, brillo {b:.3f} "
        f"(percentil {(inc['brillo'] < b).mean() * 100:.0f} en prueba)")
(RESULTADOS / "analisis_prueba.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
