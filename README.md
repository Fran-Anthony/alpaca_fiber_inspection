# Sistema de Inspección de Fibra de Alpaca

Aplicación de escritorio en Python (**PyQt5 + OpenCV + Ultralytics YOLO + Matplotlib + PySerial**) diseñada con una interfaz sobria de consola industrial para la inspección óptica de fibra de alpaca en tiempo real y el control simultáneo de un motor mediante **Arduino / L298N**.

---

## Características principales

### 1. Soporte multi-etapa con carga de modelos bajo demanda
| Etapa | Pesos | Visualización |
| :--- | :--- | :--- |
| Escarmenado y Lavado | `bestS.pt` (`yolo11n_exp2_aug`) | Cajas de impurezas con tiling |
| Cardado | `bestC.pt` | Máscaras y cajas |
| Hilado | `bestH.pt` | Máscaras y cajas |

La aplicación carga el modelo al usar cada etapa. Escarmenado y Lavado comparten la misma instancia.

### 2. Parámetros de inferencia

- **Escarmenado y Lavado** usan recortes RAW de 640 × 640 px, con 128 px de solape nominal. Un fotograma de 2448 × 2048 genera 20 tiles. Ambos umbrales empiezan en `0.10` y se pueden ajustar por clase. **Cajas** muestra partícula en rojo y paja en azul. **Auditoría** muestra la cuadrícula, las confianzas y las fusiones.
- **Cardado e Hilado** usan la imagen completa con controles de confianza y tamaño de entrada.
- Para usar otro checkpoint compatible o un motor TensorRT de 640 × 640, define `ALPACA_TILING_WEIGHTS` con la ruta del archivo.

### 3. Diagnóstico de calidad y registro de eventos (OJO)
- **Semáforo de estado de la fibra**:
  - `0` a `4` detecciones: **FIBRA LIMPIA**
  - `5` a `20` detecciones: **FIBRA CON IMPUREZAS**
  - Más de `20` detecciones: **FIBRA SUCIA**
- **Medidor horizontal (Gauge)**: escala visual de `0` a `60` con lectura numérica integrada.
- **Consola integrada (`StreamToLogger`)**: redirige `stdout` y `stderr` al panel **Registro de eventos**, mostrando tiempos de preprocesamiento, inferencia y postprocesamiento por frame.



### 4. Compatibilidad multiplataforma y utilidades conservadas
- Orden de importaciones adaptado por sistema operativo:
  - **Windows**: importa `torch` y `ultralytics` antes de `PyQt5` y `cv2` para reducir conflictos de DLL (`c10.dll`).
  - **Linux / Jetson**: importa `PyQt5` antes de `cv2` y `torch` para evitar conflictos TLS/OpenGL.


---

## Estructura de la carpeta

```text
proyecto/
├── main_usb_camera.py     # Inicio de la aplicación Qt
├── infer_tiling.py        # Inicio de la CLI
├── alpaca_inspection/     # Interfaz, cámara, motor, calibración e inferencia
├── models/               # Pesos YOLO de todas las etapas y bestE.pt de referencia
├── assets/               # Logos de la interfaz
├── reference/
│   ├── camera/           # Aplicaciones anteriores y pruebas de MindVision
│   └── tiling/           # Agents_tiling.md y herramientas de dataset Roboflow
├── runtime/              # Datos de cámara y calibración locales, ignorados por Git
└── requirements.txt
```

Ejecuta las herramientas de dataset como módulos desde la raíz. Por ejemplo:

```bash
.venv/bin/python -m reference.tiling.tiling_fibra_roboflow --help
.venv/bin/python -m reference.tiling.untiling_fibra_roboflow --help
.venv/bin/python -m reference.tiling.visualizar_untiling_coco --help
```

Las utilidades históricas de cámara permanecen en `reference/camera/`. Para ejecutarlas en este equipo, incluye `alpaca_inspection/` en `PYTHONPATH` para que encuentren el módulo del SDK.

## Instalación y ejecución

### Cámara MindVision en el Jetson

La aplicación usa MVSDK de forma predeterminada. El SDK nativo (`libMVSDK.so` en Linux)
debe estar instalado y su módulo Python `mvsdk.py` debe estar disponible en
`PYTHONPATH` o en `alpaca_inspection/mvsdk.py`. En este equipo el enlace apunta
al módulo usado por la aplicación anterior.

Ejecuta la aplicación con el entorno virtual configurado en el Jetson:

```bash
.venv/bin/python main_usb_camera.py
```

Para una cámara UVC convencional, selecciona el backend de OpenCV:

```bash
ALPACA_CAMERA_BACKEND=uvc .venv/bin/python main_usb_camera.py
```

### Opción 1: Con `uv`

`main_usb_camera.py` ya declara sus dependencias en el bloque `# /// script` al inicio del archivo. Solo ejecuta:

```bash
uv run main_usb_camera.py
```

Si prefieres ejecutarlo usando explícitamente `requirements.txt` con `uv`:

```bash
uv run --with-requirements requirements.txt python main_usb_camera.py
```

### Opción 2: Con `pip` tradicional

```bash
pip install -r requirements.txt
python main_usb_camera.py
```

---

## Guía rápida de uso

1. **Seleccionar etapa del proceso**: elige **Escarmenado**, **Lavado**, **Cardado** o **Hilado** en el panel derecho.
2. **Ajustar parámetros**: ajusta los umbrales de confianza y los controles disponibles para la etapa.
3. **Inspección en tiempo real**:
   - Pulsa **Iniciar detección** para abrir la cámara USB y comenzar la inferencia.
   - Pulsa **Detener detección** para liberar la cámara.
4. **Prueba con imágenes (sin cámara)**:
   - Pulsa **Probar imagen...** y elige un archivo (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
   - Cambia de etapa o ajusta sus controles para recalcular la predicción. Para las etapas con tiling, la imagen debe medir al menos 640 px de ancho y alto.
   - Pulsa **Limpiar** para vaciar el visor.
5. **Control de motor (Arduino / L298N)**:
   - Pulsa **Buscar** para listar los puertos disponibles y luego **Conectar**.
   - Usa **Iniciar motor**, **Detener motor**, **Adelante**, **Reversa** y el control deslizante de **Velocidad (PWM)**.

---

## Inferencia por tiling desde la terminal

```bash
.venv/bin/python infer_tiling.py --images /ruta/a/imagen.png --output resultados --save-vis
.venv/bin/python infer_tiling.py --stream mindvision
```

`--images` también admite una carpeta plana o una carpeta con sesiones `S-*/images/FLA_*.png`. Genera `predicciones.coco.json` y `resumen.csv` en una carpeta nueva. `--save-vis` añade las vistas de cajas y auditoría. `--stream` admite archivos de video, índices de cámara OpenCV y URL RTSP. Para ver los parámetros de ROI, lote y fusión, ejecuta `infer_tiling.py --help` con el mismo entorno.
