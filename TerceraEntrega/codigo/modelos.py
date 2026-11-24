"""Carga y aplicación de cada versión del clasificador (base, V1, V2, V3) y Grad-CAM.

Lo usan 11_evaluacion_prueba.py, 12_figuras_final.py y predecir.py, para que la inferencia sobre
prueba o sobre una imagen suelta use exactamente el mismo preprocesamiento y los mismos pesos.

Versiones:
  base: 8 características de color/palidez + StandardScaler + LR balanceada (Entrega 2)
  v1:   ResNet-18 congelada (ImageNet, 512 rasgos) + StandardScaler + LR balanceada C=1
  v2:   igual que v1, con C elegido por validación cruzada dentro de entrenamiento
  v3:   ResNet-18 con layer4 y capa final ajustadas (ajuste fino)
"""
import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from comun import MODELOS, extraer_color

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESV_IMAGENET = [0.229, 0.224, 0.225]
DESCRIPCION = {
    "base": "Línea base: 8 rasgos de color y palidez + LR balanceada (C=1)",
    "v1": "V1: ResNet-18 congelada (512 rasgos) + LR balanceada (C=1)",
    "v2": "V2: ResNet-18 congelada + LR balanceada (C por CV en entrenamiento)",
    "v3": "V3: ResNet-18 con ajuste fino (layer4 + capa final)",
}


def dispositivo(cfg):
    d = cfg["cnn"].get("dispositivo", "cpu")
    if d == "mps" and not torch.backends.mps.is_available():
        d = "cpu"
    return torch.device(d)


def transformacion_eval(cfg):
    """Preprocesamiento congelado de V1-V3: RGB, 224x224 bilineal, normalización de ImageNet."""
    n = cfg["cnn"]["img_size"]
    return transforms.Compose([
        transforms.Resize((n, n), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET),
    ])


def transformacion_aumento(cfg):
    """Aumento de datos de V3 (solo entrenamiento)."""
    n = cfg["cnn"]["img_size"]
    a = cfg["ajuste_fino"]["aumento"]
    pasos = [transforms.RandomResizedCrop(n, scale=tuple(a["escala_recorte"]), ratio=(0.95, 1.05),
                                          interpolation=transforms.InterpolationMode.BILINEAR)]
    if a["flip_horizontal"]:
        pasos.append(transforms.RandomHorizontalFlip())
    pasos += [transforms.RandomRotation(a["rotacion_grados"]),
              transforms.ColorJitter(brightness=a["brillo"], contrast=a["contraste"]),
              transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET)]
    return transforms.Compose(pasos)


def abrir(ruta):
    with Image.open(ruta) as im:
        return im.convert("RGB")


def resnet18_imagenet(cfg):
    pesos = getattr(models.ResNet18_Weights, cfg["cnn"]["pesos"])
    return models.resnet18(weights=pesos)


def extractor_cnn(cfg):
    """ResNet-18 de ImageNet sin la capa final: devuelve el vector de 512 de avgpool."""
    m = resnet18_imagenet(cfg)
    m.fc = nn.Identity()
    return m.eval().to(dispositivo(cfg))


def red_v3(cfg, pesos_imagenet=True):
    m = resnet18_imagenet(cfg) if pesos_imagenet else models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


@torch.no_grad()
def rasgos_cnn(rutas, cfg, extractor=None):
    extractor = extractor or extractor_cnn(cfg)
    tf, dev, b = transformacion_eval(cfg), dispositivo(cfg), cfg["cnn"]["batch"]
    salida = []
    for i in range(0, len(rutas), b):
        x = torch.stack([tf(abrir(r)) for r in rutas[i:i + b]]).to(dev)
        salida.append(extractor(x).cpu().numpy())
    return np.concatenate(salida).astype(np.float64)


@torch.no_grad()
def puntuar_v3(rutas, cfg, red):
    tf, dev, b = transformacion_eval(cfg), dispositivo(cfg), cfg["cnn"]["batch"]
    red = red.eval().to(dev)
    salida = []
    for i in range(0, len(rutas), b):
        x = torch.stack([tf(abrir(r)) for r in rutas[i:i + b]]).to(dev)
        salida.append(torch.sigmoid(red(x)).squeeze(1).cpu().numpy())
    return np.concatenate(salida).astype(np.float64)


class Version:
    """Una versión congelada del clasificador: rutas de imagen -> puntuación de glaucoma en [0, 1]."""

    def __init__(self, nombre, cfg):
        self.nombre, self.cfg = nombre, cfg
        self.descripcion = DESCRIPCION[nombre]
        # Los .joblib los genera este mismo código (04 y 08); no cargar archivos de terceros.
        if nombre == "base":
            self.lr = joblib.load(MODELOS / "modelo_base.joblib")
        elif nombre in ("v1", "v2"):
            self.lr = joblib.load(MODELOS / f"modelo_{nombre}.joblib")
            self.red = extractor_cnn(cfg)
        elif nombre == "v3":
            self.red = red_v3(cfg, pesos_imagenet=False)
            self.red.load_state_dict(torch.load(MODELOS / "modelo_v3.pt", map_location="cpu", weights_only=True))
            self.red = self.red.eval().to(dispositivo(cfg))
        else:
            raise ValueError(nombre)

    def puntuar(self, rutas):
        rutas = [str(r) for r in rutas]
        if self.nombre == "base":
            X = np.array([extraer_color(r, self.cfg) for r in rutas])
            return self.lr.predict_proba(X)[:, 1]
        if self.nombre in ("v1", "v2"):
            return self.lr.predict_proba(rasgos_cnn(rutas, self.cfg, self.red))[:, 1]
        return puntuar_v3(rutas, self.cfg, self.red)

    def gradcam(self, ruta):
        """Mapa Grad-CAM (224x224, [0,1]) del logit de glaucoma sobre layer4. None para la base."""
        if self.nombre == "base":
            return None
        red = self.red.cpu()
        activ = {}
        h = red.layer4.register_forward_hook(lambda m, i, o: activ.__setitem__("A", o))
        x = transformacion_eval(self.cfg)(abrir(ruta)).unsqueeze(0).requires_grad_(False)
        torch.use_deterministic_algorithms(False)
        try:
            salida = red(x)
            if self.nombre == "v3":
                logit = salida[0, 0]
            else:  # LR sobre los 512 rasgos estandarizados, escrita en torch para poder derivar
                sc, lr = self.lr.named_steps["scaler"], self.lr.named_steps["lr"]
                z = (salida[0] - torch.tensor(sc.mean_, dtype=torch.float32)) / torch.tensor(
                    sc.scale_, dtype=torch.float32)
                logit = (z * torch.tensor(lr.coef_[0], dtype=torch.float32)).sum() + float(lr.intercept_[0])
            A = activ["A"]
            g = torch.autograd.grad(logit, A)[0]
            cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * A).sum(1))[0]
            cam = nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                                            align_corners=False)[0, 0]
            cam = cam / cam.max() if cam.max() > 0 else cam
        finally:
            h.remove()
            torch.use_deterministic_algorithms(True)
            self.red = red.to(dispositivo(self.cfg))
        return cam.detach().numpy()
