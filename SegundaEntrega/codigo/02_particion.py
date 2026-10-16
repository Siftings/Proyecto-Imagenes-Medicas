"""Partición estratificada: validación = 20 % de ORIGA/train; prueba = ORIGA/test.

Salida: resultados/particion.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from comun import RESULTADOS, cargar_config, codigo_de

cfg = cargar_config()
data_dir = cfg["data_dir"]

train = pd.read_csv(data_dir / "ORIGA_train.csv")
test = pd.read_csv(data_dir / "ORIGA_test.csv")

ent, val = train_test_split(
    train, test_size=cfg["val_size"], stratify=train["glaucoma"], random_state=cfg["seed"]
)

partes = []
for df, nombre in [(ent, "entrenamiento"), (val, "validacion"), (test, "prueba")]:
    partes.append(pd.DataFrame({
        "codigo": df["ImageName"].map(codigo_de),
        "archivo": df["ImageName"],
        "particion": nombre,
        "etiqueta": df["glaucoma"].astype(int),
    }))
particion = pd.concat(partes, ignore_index=True)
particion.to_csv(RESULTADOS / "particion.csv", index=False)

tabla = pd.crosstab(particion["particion"], particion["etiqueta"], margins=True, margins_name="total")
print(tabla)

# Conteos que cita el informe
esperado = {"entrenamiento": (94, 269), "validacion": (23, 68), "prueba": (51, 145)}
for p, (pos, neg) in esperado.items():
    obt = (tabla.loc[p, 1], tabla.loc[p, 0])
    if obt != (pos, neg):
        print(f"AVISO: {p} da {obt[0]}/{obt[1]} y el informe usa {pos}/{neg}; hay que actualizar el informe.")
print("Conteos verificados contra el informe." if all(
    (tabla.loc[p, 1], tabla.loc[p, 0]) == v for p, v in esperado.items()) else "")
