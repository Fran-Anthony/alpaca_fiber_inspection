# CONTEXTO_TILING.md — Integración de Detección de Impurezas con Tiling (Fibra de Alpaca)

> **Propósito:** Proveer contexto técnico completo para implementar el módulo de inferencia por *tiling* (recorte en memoria + inferencia en lote + reconstrucción/deduplicación cross-tile + visualización/tiempo real) de forma 100% consistente con los scripts existentes del repositorio: `tiling_fibra_roboflow.py`, `untiling_fibra_roboflow.py` y `visualizar_untiling_coco.py`.

---

## 1. Alcance y Unificación de Dominio

* **Unificación Escarmenado + Lavado:** Ambas etapas utilizan **exactamente el mismo modelo entrenado** (`yolo11n_exp2_aug`). No se mantienen pipelines ni pesos separados entre escarmenado y lavado.
* **Arquitecturas descartadas:** Quedan descartados los modelos heredados de YOLOv7 / Cotton-YOLO (entrenados en un repositorio antiguo con clases invertidas y menor rendimiento en validación real) y las variantes más pesadas (`yolo11m`, `yolov8n`, `yolov8s`).
* **Objetivo industrial:** Priorizar la detección de impurezas (**Recall / F2-score**) en imágenes de alta resolución (`2448x2048` px típico) manteniendo latencia viable para operación en tiempo real.

---

## 2. Modelo Seleccionado, Entorno y Hardware

| Parámetro | Valor |
| :--- | :--- |
| **Modelo elegido** | `yolo11n_exp2_aug` (Ultralytics YOLO11n) |
| **Pesos (`.pt`)** | `/data/cite/Lavado/runs/ablation/exp2_aug_online/weights/best.pt` |
| **Framework / Runtime** | Ultralytics `8.4.215`, PyTorch `2.11.0+cu128`, Python `3.11.18` |
| **Entorno virtual** | `/home/cite/.venvs/lavado` |
| **Hardware de despliegue** | NVIDIA GeForce RTX 5080 (16 GB VRAM), Intel Xeon Silver 4114, 62 GB RAM, Ubuntu 22.04.5 LTS |
| **Entrada del modelo (`imgsz`)** | `640x640` nativa del tile (**sin reescalado**) |
| **Latencia medida (RTX 5080)** | `0.71–1.66 ms/tile` en PyTorch (`600–1400 tiles/s`); `<1 ms/tile` esperado en TensorRT FP16 |

### Receta de entrenamiento (referencia)
* **Base:** `yolo11n.pt`, `imgsz=640`, `batch=64`, `epochs=300` (convergencia época `163`, `patience=80`), `cls=2.0`, `seed=42`, `cache=ram`.
* **Datos:** Tiles crudos (`RAW`) de `640x640`, `overlap=128`, `resize=False`.
* **Aumentación online:** `flipud=0.5`, `fliplr=0.5`, `degrees=15.0`, `scale=0.35`, `translate=0.1`, `hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.3`, `erasing=0.0`, `mosaic=1.0`, `close_mosaic=20`.

---

## 3. Mapeo de Clases y Convenciones Visuales (CRÍTICO)

El modelo usa estrictamente `nc: 2` con el siguiente mapeo:

| `cls_id` | Nombre | Descripción | Vista Final (`RGB`, PIL) | Vista Final (`BGR`, OpenCV) | Vista Auditoría (`RGB`) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **`0`** | `particle` | Partícula / suciedad | Rojo `(255, 0, 0)` | Rojo `(0, 0, 255)` | Cian `(0, 210, 255)` |
| **`1`** | `straw` | Pajilla / paja | Azul `(0, 0, 255)` | Azul `(255, 0, 0)` | Naranja `(255, 145, 0)` |

* **Regla de color:** Verificar siempre si el canvas está en `RGB` (Pillow) o `BGR` (OpenCV) antes de dibujar. Nunca invertir `0 ↔ 1`.
* **Vista final limpia (`modelo_final`):** Debe replicar `visualizar_untiling_coco.py`: dimensión original exacta, sin encabezados, sin cuadrícula y sin texto; solo rectángulos de grosor `line_width=3` (Rojo=`0: particle`, Azul=`1: straw`).
* **Vista de auditoría (`auditoria`):** Cuadrícula de tiles `(x0, y0)`, etiquetas de clase + confianza, borde verde neón `(50, 255, 120)` con marca `[xN]` para cajas fusionadas entre `N` tiles y barra superior de resumen.

---

## 4. Evidencia Experimental y Decisiones Congeladas

### 4.1 Prohibido aplicar preprocesamiento (CLAHE / sharpening / normalización de brillo)
En la ablación sobre fibra lavada (`873` tiles de test), el preprocesamiento determinístico LAB-CLAHE + normalización + sharpening empeoró el rendimiento respecto a imágenes crudas:

| Experimento | Datos | P (%) | R (%) | F1 (%) | F2 (%) | mAP50 | `particle` AP50 | `straw` AP50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`exp2_aug_online` (GANADOR)** | RAW | **54.04** | 44.83 | **49.01** | **46.41** | **46.14** | **37.06** | **55.23** |
| `exp0_raw_640` | RAW | 53.55 | **44.85** | 48.82 | 46.36 | 45.01 | 36.54 | 53.48 |
| `exp3_prep_aug` | CLAHE | 52.62 | 43.06 | 47.36 | 44.68 | 43.74 | 34.72 | 52.76 |
| `exp1_prep_clahe` | CLAHE | 48.57 | 43.98 | 46.16 | 44.83 | 42.19 | 32.73 | 51.64 |

* **Motivo técnico:** CLAHE y sharpening amplifican microsombras de la fibra blanca esponjosa generando falsos positivos. Los tiles deben entrar **crudos (RAW)** al modelo.

### 4.2 Benchmark en Golden Test Set (200 tiles curados manualmente, 869 instancias: 562 `straw`, 307 `particle`)

| Modelo | Framework | Img | P (%) | R (%) | F1 (%) | F2 (%) | mAP50 | mAP50-95 | Inf (ms) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo11n_exp2_aug` (ELEGIDO)** | Ultralytics | 640 | 58.43 | **51.44** | **54.71** | **52.70** | 54.05 | 19.86 | **1.66** | **602.9** |
| `yolo11m_base800` | Ultralytics | 800 | **60.14** | 50.00 | 54.61 | 51.75 | **54.13** | **21.27** | 10.23 | 97.7 |
| `ft_yolov7base_mix` | YOLOv7 | 640 | 59.80 | 50.00 | 54.46 | 51.69 | 52.10 | 17.80 | 7.20 | 138.9 |
| `yolo11n_exp0_raw` | Ultralytics | 640 | 55.89 | 49.53 | 52.52 | 50.68 | 51.78 | 19.94 | 1.70 | 588.4 |
| `ft_yolov7base_lavado` | YOLOv7 | 640 | 44.20 | 44.30 | 44.25 | 44.28 | 37.80 | 12.60 | 6.50 | 153.8 |
| `yolov7base_zeroshot` | YOLOv7 | 640 | 22.10 | 11.30 | 14.95 | 12.52 | 5.12 | 1.29 | 7.00 | 142.9 |

Métricas por clase de `yolo11n_exp2_aug` (punto F1 estándar):
* `particle`: P = `57.97%`, R = `39.54%`, F1 = `47.01%`, F2 = `42.22%`, AP50 = `45.47%`
* `straw`: P = `58.88%`, R = `63.35%`, F1 = `61.03%`, F2 = `62.40%`, AP50 = `62.63%`

---

## 5. Calibración de Umbrales de Confianza (CRÍTICO)

El dataset de entrenamiento presentaba omisiones humanas parciales, lo que produjo **compresión de confianza**: el modelo localiza bien impurezas reales, pero con puntajes conservadores (`0.10–0.25`). **Nunca usar `conf=0.25` ni `conf=0.40` por defecto.**

Barrido real de `yolo11n_exp2_aug` sobre el Golden Test Set (`IoU = 0.45`):

| `conf` | All P (%) | All R (%) | All F2 (%) | `particle` P / R (%) | `straw` P / R (%) | Agnóstico P / R (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.05` | 33.27 | 74.80 | 59.85 | 33.57 / 62.21 | 33.14 / 81.67 | 35.16 / 79.06 |
| **`0.10`** | **46.27** | **68.47** | **62.47** | **46.43 / 55.05** | **46.20 / 75.80** | **49.30 / 72.96** |
| **`0.15`** | **54.03** | **60.87** | **59.37** | **54.83 / 46.25** | **53.75 / 68.86** | **57.51 / 64.79** |
| `0.20` | 62.30 | 54.78 | 56.13 | 64.36 / 39.41 | 61.63 / 63.17 | 65.97 / 58.00 |
| `0.25` | 69.88 | 48.33 | 51.51 | 74.80 / 30.94 | 68.57 / 57.83 | 73.71 / 50.98 |
| `0.30` | 77.32 | 41.20 | 45.44 | 84.62 / 25.08 | 75.54 / 50.00 | 81.86 / 43.61 |
| `0.40` | 84.31 | 24.74 | 28.81 | 90.00 / 11.73 | 83.26 / 31.85 | 87.45 / 25.66 |

### Perfiles de umbral recomendados para exponer en CLI/API
1. **Óptimo F2 (default):** `conf_particle = 0.10`, `conf_straw = 0.10` → Máximo F2 (`62.47%`), Recall global `68.47%` (`72.96%` agnóstico).
2. **Balanceado por clase:** `conf_particle = 0.10`, `conf_straw = 0.15` → Mantiene sensibilidad en `particle` (`55.05% R`) y eleva precisión en `straw` (`53.75% P`, `68.86% R`).
3. **Equilibrado (>50% Precisión):** `conf_particle = 0.15`, `conf_straw = 0.15` → `60.87% R`, `54.03% P`.
4. **Regla de implementación:** Ejecutar la predicción por lote con `base_conf = min(conf_particle, conf_straw)` (ej. `0.08–0.10`), proyectar a coordenadas globales, deduplicar y aplicar el filtro por clase al final. Soportar bandera opcional `--class-agnostic` para fusionar detecciones sin distinguir clase cuando solo importe el conteo total de impurezas.

---

## 6. Contrato Geométrico de Tiling y Deduplicación

La inferencia debe replicar exactamente la política de ventanas de `tiling_fibra_roboflow.py`:

```python
def positions(length: int, size: int = 640, overlap: int = 128) -> list[int]:
    if length < size:
        raise ValueError(f"Dimensión {length} menor que tile {size}; no se rellena ni escala.")
    result = list(range(0, length - size + 1, size - overlap))
    if result[-1] != length - size:
        result.append(length - size)
    return result
```

* **Constantes de geometría:** `tile_size = 640`, `overlap = 128`, `stride = 512`, `edge_policy = shift_last_window_to_edge`, `resize = False`.
* **Resolución de captura estándar (`2448x2048`):**
  * `xs = [0, 512, 1024, 1536, 1808]` (5 columnas; solapamiento en borde derecho = `368 px`).
  * `ys = [0, 512, 1024, 1408]` (4 filas; solapamiento en borde inferior = `256 px`).
  * Total por frame completo: **`20` tiles** (`5 x 4`).
* **Proyección global:** Sumar `(x0, y0)` a cada caja local `[x1, y1, x2, y2]` del tile y aplicar *clip* a `[0, W] x [0, H]`.
* **Deduplicación cross-tile (dos modos seleccionables):**
  1. **`cross_tile_merge` (por defecto para imágenes/auditoría, basado en `untiling_fibra_roboflow.py`):** Agrupa únicamente detecciones provenientes de **tiles distintos** (`tile_id_i != tile_id_j`) de la misma clase (o cualquier clase si `--class-agnostic`) cuando `IoU >= 0.40` **o** `IoMin >= 0.75` (`IoMin = inter / min(area1, area2)` para absorber fragmentos partidos en los bordes). Modos de fusión:
     * `wbf`: promedio ponderado por `score` (confianza), conservando el `max(score)` o promedio ponderado como confianza resultante.
     * `union`: envolvente mínima del cluster.
     * `largest`: caja de mayor área (o mayor confianza).
  2. **`batched_nms` (para máxima velocidad en tiempo real):** `torchvision.ops.batched_nms(boxes, scores, classes, iou_threshold=0.45)` directamente sobre tensores CUDA (si `--class-agnostic`, usar `torchvision.ops.nms`).

---

## 7. Inventario de Scripts Existentes en el Repositorio

| Archivo | Función actual | Qué reutilizar / respetar |
| :--- | :--- | :--- |
| `tiling_fibra_roboflow.py` | Genera tiles PNG en disco + `manifest_tiles.csv` + `tiling_config.json` desde `S-*/images/FLA_*.png`. | Reutilizar exactamente `positions(length, size, overlap)` y los defaults `tile_size=640`, `overlap=128`. En inferencia **NO escribir tiles a disco**. |
| `untiling_fibra_roboflow.py` | Traduce anotaciones COCO de tiles a imagen original y fusiona duplicados cross-tile (`IoU` + `IoMin`). | Reutilizar `bbox_iou_and_iomin`, `clip_bbox` y la lógica de componentes conectados cross-tile extendida para llevar `score` y `wbf` por confianza. |
| `visualizar_untiling_coco.py` | Genera vistas `auditoria/` y `modelo_final/` a partir de JSON COCO reconstruido. | Mantener compatibilidad total con su esquema JSON (`images`, `annotations`, `categories`, `extra.tiles`, `extra.merged_count`) y su paleta de colores. |

---

## 8. Arquitectura Solicitada para Codex

Implementar dos piezas limpias y desacopladas:

1. **`tiler_detector.py` (módulo reutilizable):**
   * Clase `TiledImpurityDetector` con carga transparente de `.pt` o TensorRT `.engine`, soporte FP16 (`half=True`), *warm-up* inicial de 1–2 lotes vacíos y caché de la grilla `coords = [(x, y) ...]` por `(H, W)` / `roi`.
   * Recorte en memoria mediante vistas NumPy (`frame[y:y+640, x:x+640]`, **sin SAHI y sin I/O en disco**).
   * Inferencia en lote (`model.predict(tiles, imgsz=640, conf=base_conf, iou=tile_iou, half=True, verbose=False)`) con soporte de *chunking* (`tile_batch_size`, default `20`) y padding automático si un `.engine` tiene batch estático mayor al número de tiles de la ROI.
   * Soporte opcional de ROI (`roi=(x1, y1, x2, y2)`) aplicada antes de calcular la grilla, sumando `(roi_x1, roi_y1)` al proyectar cajas de vuelta al frame original.
   * Método `predict_frame(frame_bgr_or_rgb)` que devuelva detecciones estructuradas (`bbox_xyxy`, `score`, `class_id`, `class_name`, `merged_count`, `source_tiles`) y métricas de tiempo (`tiling_ms`, `infer_ms`, `postprocess_ms`, `total_ms`).

2. **`infer_tiling.py` (CLI unificado):**
   * **Modo imágenes / lote:** Acepta una imagen individual, una carpeta plana o la estructura `S-*/images/FLA_*.png`. Exporta:
     * JSON en formato COCO (compatible con `visualizar_untiling_coco.py`) con `score` y metadatos `extra`.
     * CSV resumen por imagen (`image`, `width`, `height`, `num_tiles`, `particle_count`, `straw_count`, `total_impurities`, `total_ms`).
     * Bandera `--save-vis` para guardar vistas `modelo_final/` y `auditoria/` directamente.
   * **Modo tiempo real (video / cámara / RTSP):** Bucle OpenCV con dibujo de cajas (`BGR`: Rojo=`particle`, Azul=`straw`), FPS reales, latencia en ms y conteo instantáneo en pantalla.

---

## 9. Exportación a TensorRT y Restricciones Conocidas

* **Exportación recomendada a TensorRT FP16 (para `2448x2048` → `20` tiles):**
  ```bash
  yolo export \
    model=/data/cite/Lavado/runs/ablation/exp2_aug_online/weights/best.pt \
    format=engine \
    imgsz=640 \
    half=True \
    batch=20 \
    device=0
  ```
  *(Si se recorta una ROI o se usa cámara 1080p, ajustar `batch=` al número exacto de tiles o exportar con `dynamic=True`.)*
* **Bug conocido del servidor (DataLoader):** Nunca usar `workers > 0` si se invoca `model.val()` de Ultralytics (provoca deadlock en `multiprocessing/resource_sharer.py`). Usar siempre `workers=0`.
* **Inmutabilidad de datos:** Ningún script debe modificar ni sobrescribir las imágenes originales ni los metadatos dentro de `S-*/`.
* **Escala óptica fija:** Como el modelo fue entrenado con `resize=False`, la distancia cámara–fibra (píxeles/mm) en producción debe mantenerse equivalente a la de adquisición.

---

## 10. Checklist de Verificación (Do / Don't)

* **DO:** Usar `yolo11n_exp2_aug`, `tile_size=640`, `overlap=128`, imágenes RAW y `conf=0.10` (o `particle=0.10`, `straw=0.15`).
* **DO:** Procesar todos los tiles de un frame en un solo batch en memoria GPU.
* **DO:** Respetar `0 = particle` (Rojo) y `1 = straw` (Azul), cuidando `RGB` vs `BGR`.
* **DON'T:** No redimensionar `2448x2048` directo a `640x640`.
* **DON'T:** No aplicar CLAHE, sharpening ni filtros de contraste antes de inferir.
* **DON'T:** No usar SAHI en tiempo real ni guardar tiles temporales en disco.
* **DON'T:** No usar `conf=0.25` o `0.40` como default.