# Guion de la sustentación: jueves 26 de noviembre (8 min + 4 min de preguntas)

La guía exige que todos los integrantes intervengan y que cada uno pueda explicar el método, una decisión y un resultado. El reparto de abajo es una propuesta y se puede ajustar.

| Min | Diapositiva | Quién | Idea central |
|---|---|---|---|
| 0:00–0:30 | 1 Portada | Giuseppe | Presentar al equipo y la pregunta |
| 0:30–1:30 | 2 Problema y datos | Giuseppe | Entrada, salida y clase positiva. 650 imágenes, como máximo 641 ojos, personas no verificables |
| 1:30–2:45 | 3 Partición y fuga | Giuseppe | 363/91/192, semilla 42. pHash y revisión a ojo: 9 pares del mismo ojo y 4 imágenes excluidas. Sin ID de paciente queda un límite |
| 2:45–4:00 | 4 Método | Juan David | El cambio principal es la representación: V1 aísla ese efecto, V2 regulariza y V3 hace ajuste fino. La regla de selección se fijó antes de comparar |
| 4:00–5:00 | 5 Validación | Juan David | AUC de 0,667 → 0,831. V1 y V3 memorizan el entrenamiento. Se eligió V2 con umbral 0,4834 |
| 5:00–6:15 | 6 Prueba | Daniel | 40/49 y 75/143, AUC de 0,751 sobre 192 imágenes. Qué mejoró y qué empeoró |
| 6:15–7:15 | 7 Error y demostración | Daniel explica los casos; Juan David corre `predecir.py` | test/580 (FN) y test/183 (caso límite). Las causas se presentan como hipótesis |
| 7:15–8:00 | 8 Conclusión | Daniel | Hay señal, pero no alcanza para tamizaje. Qué no se puede concluir y cuál es el trabajo futuro |

**Demostración (desde `codigo/`, con el entorno activado):**
```bash
python predecir.py ../archive/ORIGA/test/580.jpg ../archive/ORIGA/test/240.jpg
```
Si el entorno falla, se muestra la salida guardada en la diapositiva 7 y se explica qué pasó.

## Respuestas a las 6 preguntas de la guía

1. **¿Qué es un caso independiente?** En principio, una persona. La copia no trae ID de paciente, así que contamos imágenes: son 650 y corresponden a lo sumo a 641 ojos distintos, porque 9 pares son el mismo ojo. Que haya una persona por partición no es verificable.
2. **¿Qué datos usamos para ajustar y cuáles para evaluar?** Todo se ajustó con las 363 imágenes de entrenamiento: estandarización, coeficientes, la elección de C (por CV dentro de entrenamiento) y los pesos de V3. La validación (91 imágenes) solo sirvió para comparar versiones y fijar el umbral. La prueba (192 imágenes) se usó una sola vez, como deja constancia `prueba_ejecutada.lock`.
3. **¿Qué cambió entre la línea base y la variante?** La representación de la imagen: se pasó de 8 medidas de color y palidez a 512 rasgos de ResNet-18 preentrenada en ImageNet. En V2 además se eligió una regularización fuerte (C = 1e-4).
4. **¿Por qué esa métrica?** En tamizaje, un falso negativo cuesta más, así que la sensibilidad mide los casos que se escapan y la especificidad las falsas alarmas. El AUC compara versiones sin depender del umbral. La exactitud engaña: el clasificador que siempre responde "sano" llega a 0,747.
5. **¿Qué error observamos y qué evidencia apoya la explicación?** test/580 es un FN con p = 0,420: el disco está descentrado y el Grad-CAM cae en el borde, no en la copa. Es una hipótesis. Se podría comprobar recentrando el recorte y volviendo a puntuar sin reentrenar.
6. **¿Cómo se reproduce una predicción?** Se sigue el README: se instalan las dependencias, se ubica `archive/` y se corre `python predecir.py <imagen>` o los scripts 10–13. Los pesos van incluidos y en CPU los resultados salen idénticos.
