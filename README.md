# Sistema de Inspección de Fibra de Alpaca

Aplicación de escritorio en Python (**PyQt5 + OpenCV + Ultralytics YOLO + Matplotlib + PySerial**) diseñada con una interfaz sobria de consola industrial para la inspección óptica de fibra de alpaca en tiempo real y el control simultáneo de un motor mediante **Arduino / L298N**.

---

## Características principales

### 1. Soporte multi-etapa con carga de modelos bajo demanda
El sistema permite alternar entre las tres etapas del proceso transformativo de la fibra sin cargar todos los pesos en memoria al inicio:

| Etapa | Archivo de pesos | Tipo de tarea | Visualización |  |
| :--- | :--- | :--- | :--- | :--- |
| **Escarmenado** | `bestE.pt` | Detección | Bounding boxes + etiquetas 
| **Cardado** | `bestC.pt` | Detección | Bounding boxes + etiquetas |  |
| **Hilado** | `bestH.pt` | Segmentación | Máscara  + contorno + bounding box + etiquetas |  |

- Al cambiar de etapa, el modelo anterior se libera de memoria (`del`, `gc.collect()` y `torch.cuda.empty_cache()` si hay GPU CUDA) antes de cargar el nuevo.
- Es posible cambiar de etapa incluso con la cámara en funcionamiento.

### 2. Parámetros de inferencia graduables (compartidos)
- **Confianza (`conf`)**: ajustable entre `0.05` y `0.95` (paso de `0.05`, valor inicial `0.50`).
- **Tamaño de entrada (`imgsz`)**: seleccionable entre `320`, `416`, `512`, `640`, `800`, `960`, `1024`, `1280` y `1600` (valor inicial `416`).
  - Valores más bajos priorizan velocidad (FPS) en tiempo real.
  - Valores más altos conservan más detalle fino para impurezas pequeñas o máscaras de segmentación en fotos de alta resolución.

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

Coloca todos los archivos en la misma carpeta del proyecto:

```text
proyecto/
├── main_usb_camera.py   # Aplicación principal
├── mindvision_video.py  # Captura MindVision mediante MVSDK
├── requirements.txt     # Dependencias de Python
├── README.md            # Documentación
├── bestE.pt             # Pesos YOLO - Escarmenado (detección)
├── bestC.pt             # Pesos YOLO - Cardado (detección)
├── bestH.pt             # Pesos YOLO - Hilado (segmentación)
├── logo_left.png        # Opcional - Logo institucional izquierdo
└── logo_right.png       # Opcional - Logo institucional derecho
```




---

## Instalación y ejecución

### Cámara MindVision en el Jetson

La aplicación usa MVSDK de forma predeterminada. El SDK nativo (`libMVSDK.so` en Linux)
debe estar instalado y su módulo Python `mvsdk.py` debe estar disponible en la carpeta
del proyecto o en `PYTHONPATH`. En este equipo, el enlace de la carpeta del proyecto
apunta al módulo usado por la aplicación anterior.

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

1. **Seleccionar etapa del proceso**: haz clic en **Escarmenado**, **Cardado** o **Hilado** en el panel derecho. El modelo correspondiente (`bestE.pt`, `bestC.pt` o `bestH.pt`) se cargará automáticamente.
2. **Ajustar parámetros**: modifica **Confianza** e **Imgsz** según la sensibilidad y resolución deseadas.
3. **Inspección en tiempo real**:
   - Pulsa **Iniciar detección** para abrir la cámara USB y comenzar la inferencia.
   - Pulsa **Detener detección** para liberar la cámara.
4. **Prueba con imágenes (sin cámara)**:
   - Pulsa **Probar imagen...** y elige un archivo (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
   - Cambia de etapa o ajusta `Confianza` / `Imgsz` para ver la actualización inmediata.
   - Pulsa **Limpiar** para vaciar el visor.
5. **Control de motor (Arduino / L298N)**:
   - Pulsa **Buscar** para listar los puertos disponibles y luego **Conectar**.
   - Usa **Iniciar motor**, **Detener motor**, **Adelante**, **Reversa** y el control deslizante de **Velocidad (PWM)**.

---
