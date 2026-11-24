"""Variante V3: ajuste fino de ResNet-18 (layer4 + capa final) con aumento de datos.

Entrena solo con las 363 de entrenamiento, durante un número FIJO de épocas (sin early stopping
sobre validación), y evalúa una vez sobre las 91 de validación. La prueba no se toca.
Tarda unos minutos en CPU; 11, 12 y predecir.py usan los pesos guardados sin reentrenar.
Salidas: modelos/modelo_v3.pt, resultados/historial_v3.csv,
         resultados/predicciones_validacion_v3.csv, resultados/metricas_validacion_v3.json
"""
import time

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from comun import MODELOS, RESULTADOS, cargar_config, fijar_semillas, guardar_json, resumen, tabla_por_caso
from modelos import DESCRIPCION, abrir, dispositivo, puntuar_v3, red_v3, transformacion_aumento

cfg = cargar_config()
seed = cfg["seed"]
fijar_semillas(seed)
af = cfg["ajuste_fino"]
dev = dispositivo(cfg)

part = pd.read_csv(RESULTADOS / "particion.csv")
ent = part[part["particion"] == "entrenamiento"].reset_index(drop=True)
val = part[part["particion"] == "validacion"].reset_index(drop=True)
rutas_ent = [cfg["data_dir"] / f for f in ent["archivo"]]
rutas_val = [cfg["data_dir"] / f for f in val["archivo"]]


class Fondos(Dataset):
    def __init__(self, rutas, y, tf):
        self.imgs = [abrir(r) for r in rutas]  # 363 imágenes pequeñas: caben en memoria
        self.y, self.tf = torch.tensor(y, dtype=torch.float32), tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.y[i]


gen = torch.Generator().manual_seed(seed)
cargador = DataLoader(Fondos(rutas_ent, ent["etiqueta"].to_numpy(), transformacion_aumento(cfg)),
                      batch_size=af["batch"], shuffle=True, generator=gen, num_workers=0)

red = red_v3(cfg)
for nombre, p in red.named_parameters():
    p.requires_grad = nombre.split(".")[0] in af["capas_entrenables"]
entrenables = [p for p in red.parameters() if p.requires_grad]
red = red.to(dev)

n_pos, n_neg = int(ent["etiqueta"].sum()), int((ent["etiqueta"] == 0).sum())
perdida = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=dev))  # balance de clases
opt = torch.optim.AdamW(entrenables, lr=af["lr"], weight_decay=af["weight_decay"])


def modo_entrenamiento():
    """layer4 y fc en modo train; las capas congeladas mantienen sus estadísticas de BatchNorm."""
    red.eval()
    for nombre in af["capas_entrenables"]:
        getattr(red, nombre).train()


historial = []
t0 = time.time()
for epoca in range(1, af["epocas"] + 1):
    modo_entrenamiento()
    total, n = 0.0, 0
    for x, y in cargador:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = perdida(red(x).squeeze(1), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    historial.append({"epoca": epoca, "perdida_entrenamiento": total / n})
    print(f"  época {epoca:2d}/{af['epocas']}  pérdida {total / n:.4f}  ({time.time() - t0:.0f} s)")

torch.save(red.cpu().state_dict(), MODELOS / "modelo_v3.pt")
pd.DataFrame(historial).to_csv(RESULTADOS / "historial_v3.csv", index=False)

# Una sola evaluación en validación, con los pesos de la última época
red = red.to(dev)
prob_val = puntuar_v3(rutas_val, cfg, red)
prob_ent = puntuar_v3(rutas_ent, cfg, red)
y_val = val["etiqueta"].to_numpy()
tabla_por_caso(val["codigo"], "validacion", "v3", y_val, prob_val, cfg["umbral"]).to_csv(
    RESULTADOS / "predicciones_validacion_v3.csv", index=False)
salida = {"version": "v3", "descripcion": DESCRIPCION["v3"],
          **resumen(y_val, prob_val, cfg["umbral"], cfg["n_bootstrap"], seed),
          "AUC_entrenamiento": roc_auc_score(ent["etiqueta"], prob_ent),
          "parametros": af, "tiempo_entrenamiento_s": round(time.time() - t0, 1),
          "versiones": {"torch": torch.__version__}}
guardar_json(salida, RESULTADOS / "metricas_validacion_v3.json")
m = salida["metricas"]
print(f"v3: AUC val {m['AUC']['valor']:.3f} [{m['AUC']['IC95'][0]:.2f}, {m['AUC']['IC95'][1]:.2f}] | "
      f"AUC ent {salida['AUC_entrenamiento']:.3f} | sens {salida['denominadores']['sensibilidad']} "
      f"esp {salida['denominadores']['especificidad']}")
