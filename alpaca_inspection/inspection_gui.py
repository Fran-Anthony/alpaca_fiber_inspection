"""Interfaz Qt de inspección de fibra de alpaca."""
import gc
import io
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor

# PyTorch se carga antes de Qt en Windows y después en Jetson.
if platform.system() == "Windows":
    from . import inspection_inference

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QSize, pyqtSignal

from .inspection_inference import PROCESS_MODELS, YOLO, YOLO_IMPORT_ERROR, draw_custom_predictions, torch
from .camera_control import create_camera
from .color_calibration import ColorCalibration, DEFAULT_COLOR_PARAMS
from .motor_control import MotorControls
from .tiler_detector import TiledImpurityDetector, render_detections
from .paths import ASSETS_DIR, MODELS_DIR, PROJECT_DIR

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors


class StreamToLogger(io.TextIOBase):
    """
    Clase utilitaria para redirigir flujos de salida (stdout/stderr) a un widget de la GUI.
    Permite ver los print() y errores directamente en la interfaz de la aplicación.
    """

    def __init__(self, log_function):
        self.log_function = log_function
        self.line_buffer = ''

    def write(self, text):
        for char in text:
            if char == '\n':
                if self.line_buffer.strip():
                    self.log_function(self.line_buffer.strip())
                self.line_buffer = ''
            else:
                self.line_buffer += char
        return len(text)

    def flush(self):
        if self.line_buffer.strip():
            self.log_function(self.line_buffer.strip())
            self.line_buffer = ''


class MainWindow(QMainWindow, ColorCalibration, MotorControls):
    """
    Ventana principal responsiva y táctil de la aplicación de inspección de fibra de alpaca.
    Se adapta automáticamente a pantallas pequeñas (empotradas/Jetson), medianas y grandes,
    soporta desplazamiento táctil con el dedo y permite plegar el panel desde la parte superior.
    """

    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.base_dir = str(PROJECT_DIR)

        # --- Parámetros de calibración por defecto ---
        self.params = DEFAULT_COLOR_PARAMS.copy()
        self.color_fix_enabled = bool(self.params.get("color_fix_enabled", True))

        # --- Estado de Modelo (carga bajo demanda) ---
        self.model = None
        self.current_stage_key = None
        self.loaded_model_key = None
        self.infer_conf = 0.50
        self.infer_imgsz = 416
        self.last_tiled_detections = []
        self.tiled_executor = ThreadPoolExecutor(max_workers=1)
        self.tiled_job = None
        self.tiled_timer = QTimer(self)
        self.tiled_timer.setInterval(30)
        self.tiled_timer.timeout.connect(self._finish_tiled_inference)

        # --- Estado temporal para pruebas con imágenes ---
        self.test_image_bgr = None
        self.test_image_path = None
        self.last_rendered_pixmap = None

        # --- Estado responsivo y panel plegable ---
        self._layout_mode = None
        self._sidebar_visible = True
        self._saved_sidebar_width = 330
        self.logo_left_pixmap = None
        self.logo_right_pixmap = None

        # --- Configuración Inicial Adaptativa ---
        self.setWindowTitle("Sistema de Inspección de Fibra de Alpaca")
        self.setMinimumSize(640, 420)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

        self._init_ui()
        self._enable_touch_scrolling()
        self._apply_styles()

        # Atajos útiles en pantallas pequeñas empotradas
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        self.sidebar_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self.sidebar_shortcut.activated.connect(self._toggle_sidebar)

        self.cap = None

        # --- Control de motor Arduino/L298N ---
        self.serial_port = None
        self.motor_speed = 90
        self.motor_running = False
        self.motor_direction = "FORWARD"

        self.motor_serial_timer = QTimer(self)
        self.motor_serial_timer.timeout.connect(self._read_motor_serial)
        self.motor_serial_timer.start(50)

        # Timer para el loop de procesamiento de video (~30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Redirección de consola al widget de log en la GUI
        self.log_message.connect(self.log)
        sys.stdout = StreamToLogger(self.log_message.emit)
        sys.stderr = StreamToLogger(self.log_message.emit)

        # Seleccionar la primera etapa por defecto (sin cargar hasta usarla)
        self.select_process_stage("escarmenado", load_now=False)

        # Ajustar tamaño inicial según la pantalla detectada
        self._configure_initial_window_geometry()

    def _enable_touch_scrolling(self):
        """Habilita desplazamiento táctil suave con el dedo en el panel lateral y el log."""
        try:
            for viewport in (self.sidebar_scroll.viewport(), self.log_area.viewport()):
                viewport.setAttribute(Qt.WA_AcceptTouchEvents, True)
                QScroller.grabGesture(viewport, QScroller.LeftMouseButtonGesture)
                scroller = QScroller.scroller(viewport)
                props = scroller.scrollerProperties()
                props.setScrollMetric(QScrollerProperties.OvershootDragResistanceFactor, 0.35)
                props.setScrollMetric(QScrollerProperties.OvershootScrollDistanceFactor, 0.20)
                props.setScrollMetric(QScrollerProperties.DecelerationFactor, 0.65)
                scroller.setScrollerProperties(props)
        except Exception as e:
            print(f"Aviso: no se pudo activar gesto táctil en scroll: {e}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_sidebar(self):
        """Oculta o despliega el panel lateral de configuración para ampliar el visor."""
        if self._sidebar_visible:
            sizes = self.main_splitter.sizes()
            if len(sizes) == 2 and sizes[1] >= 200:
                self._saved_sidebar_width = sizes[1]
            self.sidebar_scroll.setVisible(False)
            self._sidebar_visible = False
            self.toggle_sidebar_btn.setText("❮")
            self.toggle_sidebar_btn.setToolTip("Mostrar panel de configuración (Ctrl+B)")
        else:
            self.sidebar_scroll.setVisible(True)
            self._sidebar_visible = True
            self.toggle_sidebar_btn.setText("❯")
            self.toggle_sidebar_btn.setToolTip("Ocultar panel de configuración (Ctrl+B)")
            total_w = max(self.main_splitter.width(), self.width() - 24)
            sidebar_w = max(250, min(self._saved_sidebar_width, 460))
            video_w = max(320, total_w - sidebar_w)
            self.main_splitter.setSizes([video_w, sidebar_w])

        # Reescalar imagen en el visor en el siguiente ciclo de layout
        QTimer.singleShot(0, self._render_last_pixmap)

    def _configure_initial_window_geometry(self):
        """Adapta el tamaño inicial de la ventana al monitor actual."""
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            # En pantallas pequeñas/empotradas (ej. 800x480, 1024x600, 1280x720) abrir maximizada
            if avail.width() <= 1366 or avail.height() <= 768:
                self.setGeometry(avail)
                self._apply_responsive_layout(avail.width(), avail.height())
                self.showMaximized()
                return

            # En pantallas medianas o grandes, usar 90% del área útil centrada
            w = min(int(avail.width() * 0.90), 1760)
            h = min(int(avail.height() * 0.90), 980)
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + (avail.height() - h) // 2
            self.setGeometry(x, y, w, h)
            self._apply_responsive_layout(w, h)
            self.show()
        else:
            self.resize(1280, 720)
            self._apply_responsive_layout(1280, 720)
            self.show()

    def _make_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("divider")
        return line

    def _load_logo_pixmap(self, filename):
        for path in (str(ASSETS_DIR / filename), filename):
            if os.path.exists(path):
                pm = QPixmap(path)
                if not pm.isNull():
                    return pm
        return None

    def _init_ui(self):
        """Inicializa la estructura y disposición responsiva y táctil de los widgets."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Barra Superior Plana ---
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        self.title_bar_layout = QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(16, 6, 16, 6)
        self.title_bar_layout.setSpacing(10)

        self.logo_left_pixmap = self._load_logo_pixmap("logo_left.png")
        self.logo_right_pixmap = self._load_logo_pixmap("logo_right.png")

        self.logo_left = QLabel()
        self.logo_left.setObjectName("logoBox")
        if self.logo_left_pixmap is None:
            self.logo_left.setText("Logo Izquierdo")

        self.title_label = QLabel("Sistema de Inspección de Fibra de Alpaca")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)

        self.logo_right = QLabel()
        self.logo_right.setObjectName("logoBox")
        if self.logo_right_pixmap is None:
            self.logo_right.setText("Logo Derecho")

        # Botón de pantalla completa para pantallas táctiles/empotradas sin teclado
        self.toggle_fullscreen_btn = QPushButton("⛶")
        self.toggle_fullscreen_btn.setObjectName("fullscreenBtn")
        self.toggle_fullscreen_btn.setFixedSize(38, 32)
        self.toggle_fullscreen_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_fullscreen_btn.setToolTip("Pantalla completa (F11)")
        self.toggle_fullscreen_btn.clicked.connect(self._toggle_fullscreen)

        # Pequeña barrita superior para plegar/desplegar el panel de configuración
        self.toggle_sidebar_btn = QPushButton("❯")
        self.toggle_sidebar_btn.setObjectName("toggleSidebarBtn")
        self.toggle_sidebar_btn.setFixedSize(44, 32)
        self.toggle_sidebar_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_sidebar_btn.setToolTip("Ocultar panel de configuración (Ctrl+B)")
        self.toggle_sidebar_btn.clicked.connect(self._toggle_sidebar)

        self.title_bar_layout.addWidget(self.logo_left)
        self.title_bar_layout.addStretch(1)
        self.title_bar_layout.addWidget(self.title_label, 2)
        self.title_bar_layout.addStretch(1)
        self.title_bar_layout.addWidget(self.logo_right)
        self.title_bar_layout.addWidget(self.toggle_fullscreen_btn, 0, Qt.AlignVCenter)
        self.title_bar_layout.addWidget(self.toggle_sidebar_btn, 0, Qt.AlignVCenter)

        # --- Área de Contenido Principal con QSplitter adaptable ---
        content_area = QWidget()
        content_area.setObjectName("contentArea")
        self.content_layout = QHBoxLayout(content_area)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)

        # Visor de cámara (Izquierda) - sin mínimos rígidos que rompan pantallas pequeñas
        self.video_label = QLabel("Seleccione una etapa e inicie la detección o cargue una imagen de prueba")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setWordWrap(True)
        self.video_label.setMinimumSize(320, 220)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.main_splitter.addWidget(self.video_label)

        # Panel lateral derecho con scroll suave y táctil
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(250)
        self.sidebar_scroll.setMaximumWidth(460)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setContentsMargins(14, 14, 14, 14)
        self.sidebar_layout.setSpacing(10)

        # --- Panel de Selección de Etapa / Modelo ---
        model_title = QLabel("Etapa del proceso")
        model_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(model_title)

        stage_row = QHBoxLayout()
        stage_row.setSpacing(5)
        self.stage_buttons = {}
        for key, cfg in PROCESS_MODELS.items():
            btn = QPushButton(cfg["label"])
            btn.setCheckable(True)
            btn.setObjectName("stageBtn")
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, k=key: self.select_process_stage(k, load_now=True))
            self.stage_buttons[key] = btn
            stage_row.addWidget(btn)
        self.sidebar_layout.addLayout(stage_row)

        self.model_info_label = QLabel("Modelo: sin cargar")
        self.model_info_label.setObjectName("monoMuted")
        self.model_info_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.model_info_label)

        # --- Parámetros graduables de inferencia (compartidos) ---
        self.standard_params = QWidget()
        params_row = QHBoxLayout(self.standard_params)
        params_row.setContentsMargins(0, 0, 0, 0)
        params_row.setSpacing(5)

        conf_lbl = QLabel("Conf.")
        conf_lbl.setObjectName("mutedLabel")

        self.btn_conf_minus = QPushButton("-")
        self.btn_conf_minus.setObjectName("touchStepBtn")
        self.btn_conf_minus.setFixedSize(30, 34)
        self.btn_conf_minus.setCursor(Qt.PointingHandCursor)
        self.btn_conf_minus.clicked.connect(lambda: self._step_conf(-0.05))

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.05, 0.95)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(self.infer_conf)
        self.spin_conf.setMinimumHeight(34)
        self.spin_conf.setKeyboardTracking(False)
        self.spin_conf.valueChanged.connect(self._on_infer_params_changed)

        self.btn_conf_plus = QPushButton("+")
        self.btn_conf_plus.setObjectName("touchStepBtn")
        self.btn_conf_plus.setFixedSize(30, 34)
        self.btn_conf_plus.setCursor(Qt.PointingHandCursor)
        self.btn_conf_plus.clicked.connect(lambda: self._step_conf(0.05))

        imgsz_lbl = QLabel("Imgsz")
        imgsz_lbl.setObjectName("mutedLabel")
        self.combo_imgsz = QComboBox()
        for sz in [320, 416, 512, 640, 800, 960, 1024, 1280, 1600]:
            self.combo_imgsz.addItem(str(sz), sz)
        self.combo_imgsz.setCurrentText(str(self.infer_imgsz))
        self.combo_imgsz.setMinimumHeight(34)
        self.combo_imgsz.currentIndexChanged.connect(self._on_infer_params_changed)

        params_row.addWidget(conf_lbl)
        params_row.addWidget(self.btn_conf_minus)
        params_row.addWidget(self.spin_conf, 1)
        params_row.addWidget(self.btn_conf_plus)
        params_row.addWidget(imgsz_lbl)
        params_row.addWidget(self.combo_imgsz, 1)
        self.sidebar_layout.addWidget(self.standard_params)

        self.tiled_params = QWidget()
        tiled_form = QFormLayout(self.tiled_params)
        tiled_form.setContentsMargins(0, 0, 0, 0)
        self.spin_particle = QDoubleSpinBox()
        self.spin_straw = QDoubleSpinBox()
        for spin in (self.spin_particle, self.spin_straw):
            spin.setRange(0.05, 0.95)
            spin.setSingleStep(0.05)
            spin.setValue(0.10)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._on_infer_params_changed)
        tiled_form.addRow("Conf. partícula", self.spin_particle)
        tiled_form.addRow("Conf. paja", self.spin_straw)
        self.tiled_view = QComboBox()
        self.tiled_view.addItems(["Cajas", "Auditoría"])
        self.tiled_view.currentIndexChanged.connect(self._on_infer_params_changed)
        tiled_form.addRow("Vista", self.tiled_view)
        self.tiled_summary = QLabel("Tiles de 640 px · solape de 128 px")
        self.tiled_summary.setWordWrap(True)
        tiled_form.addRow(self.tiled_summary)
        self.sidebar_layout.addWidget(self.tiled_params)

        # Botón principal de cámara en tiempo real
        self.detect_button = QPushButton("Iniciar detección")
        self.detect_button.setObjectName("detectButton")
        self.detect_button.setMinimumHeight(42)
        self.detect_button.setCursor(Qt.PointingHandCursor)
        self.detect_button.clicked.connect(self.start_detection)
        self.sidebar_layout.addWidget(self.detect_button)

        # === MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES (UI) ===
        test_img_row = QHBoxLayout()
        test_img_row.setSpacing(6)
        self.btn_test_image = QPushButton("Probar imagen...")
        self.btn_test_image.setMinimumHeight(36)
        self.btn_test_image.setCursor(Qt.PointingHandCursor)
        self.btn_test_image.clicked.connect(self._open_test_image)

        self.btn_clear_test_image = QPushButton("Limpiar")
        self.btn_clear_test_image.setMinimumHeight(36)
        self.btn_clear_test_image.setCursor(Qt.PointingHandCursor)
        self.btn_clear_test_image.clicked.connect(self._clear_test_image)
        self.btn_clear_test_image.setEnabled(False)

        test_img_row.addWidget(self.btn_test_image, 2)
        test_img_row.addWidget(self.btn_clear_test_image, 1)
        self.sidebar_layout.addLayout(test_img_row)
        # === FIN MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES (UI) ===

        self.sidebar_layout.addWidget(self._make_divider())

        # Estado de la fibra
        status_title = QLabel("Estado de la fibra")
        status_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(status_title)

        self.status_label = QLabel("ESPERANDO")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(self.status_label)

        # Nivel de impurezas
        gauge_header = QHBoxLayout()
        gauge_title = QLabel("Nivel de impurezas")
        gauge_title.setObjectName("mutedLabel")
        self.gauge_value_label = QLabel("0 / 60")
        self.gauge_value_label.setObjectName("monoValue")
        gauge_header.addWidget(gauge_title)
        gauge_header.addStretch()
        gauge_header.addWidget(self.gauge_value_label)
        self.sidebar_layout.addLayout(gauge_header)

        self.gauge_figure = Figure(figsize=(3.5, 0.5), dpi=100)
        self.gauge_figure.patch.set_facecolor("#FFFFFF")
        self.gauge_canvas = FigureCanvas(self.gauge_figure)
        self.gauge_canvas.setFixedHeight(24)
        self.sidebar_layout.addWidget(self.gauge_canvas)

        self.sidebar_layout.addWidget(self._make_divider())

        # Control de motor Arduino/L298N
        motor_title = QLabel("Control de motor")
        motor_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(motor_title)

        port_row = QHBoxLayout()
        port_row.setSpacing(5)
        self.motor_port_combo = QComboBox()
        self.motor_port_combo.setMinimumHeight(34)
        self.motor_refresh_button = QPushButton("Buscar")
        self.motor_refresh_button.setMinimumHeight(34)
        self.motor_connect_button = QPushButton("Conectar")
        self.motor_connect_button.setMinimumHeight(34)
        self.motor_refresh_button.clicked.connect(self._find_motor_ports)
        self.motor_connect_button.clicked.connect(self._toggle_motor_connection)
        port_row.addWidget(self.motor_port_combo, 1)
        port_row.addWidget(self.motor_refresh_button)
        port_row.addWidget(self.motor_connect_button)
        self.sidebar_layout.addLayout(port_row)

        self.motor_connection_label = QLabel("Arduino: desconectado")
        self.motor_connection_label.setObjectName("mutedLabel")
        self.sidebar_layout.addWidget(self.motor_connection_label)

        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.motor_start_button = QPushButton("Iniciar motor")
        self.motor_stop_button = QPushButton("Detener motor")
        self.motor_start_button.setObjectName("motorActionBtn")
        self.motor_stop_button.setObjectName("motorStopBtn")
        self.motor_start_button.setMinimumHeight(36)
        self.motor_stop_button.setMinimumHeight(36)
        self.motor_start_button.clicked.connect(self._motor_start)
        self.motor_stop_button.clicked.connect(self._motor_stop)
        run_row.addWidget(self.motor_start_button)
        run_row.addWidget(self.motor_stop_button)
        self.sidebar_layout.addLayout(run_row)

        direction_row = QHBoxLayout()
        direction_row.setSpacing(6)
        self.motor_reverse_button = QPushButton("Reversa")
        self.motor_forward_button = QPushButton("Adelante")
        self.motor_reverse_button.setMinimumHeight(36)
        self.motor_forward_button.setMinimumHeight(36)
        self.motor_reverse_button.clicked.connect(self._motor_reverse)
        self.motor_forward_button.clicked.connect(self._motor_forward)
        direction_row.addWidget(self.motor_reverse_button)
        direction_row.addWidget(self.motor_forward_button)
        self.sidebar_layout.addLayout(direction_row)

        speed_row = QHBoxLayout()
        speed_title = QLabel("Velocidad")
        speed_title.setObjectName("mutedLabel")
        self.motor_speed_label = QLabel("PWM: 90  |  35 %")
        self.motor_speed_label.setObjectName("monoValue")
        speed_row.addWidget(speed_title)
        speed_row.addStretch()
        speed_row.addWidget(self.motor_speed_label)
        self.sidebar_layout.addLayout(speed_row)

        self.motor_speed_slider = QSlider(Qt.Horizontal)
        self.motor_speed_slider.setRange(0, 255)
        self.motor_speed_slider.setValue(90)
        self.motor_speed_slider.setMinimumHeight(32)
        self.motor_speed_slider.valueChanged.connect(self._motor_speed_changed)
        self.sidebar_layout.addWidget(self.motor_speed_slider)

        self.motor_response_label = QLabel("Respuesta Arduino: sin datos")
        self.motor_response_label.setObjectName("monoMuted")
        self.motor_response_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.motor_response_label)

        self._find_motor_ports()

        self.sidebar_layout.addWidget(self._make_divider())

        # Registro de eventos
        log_title = QLabel("Registro de eventos")
        log_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(log_title)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(95)
        self.sidebar_layout.addWidget(self.log_area, 1)

        self.sidebar_scroll.setWidget(sidebar)
        self.main_splitter.addWidget(self.sidebar_scroll)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)

        self.content_layout.addWidget(self.main_splitter, 1)

        main_layout.addWidget(self.title_bar)
        main_layout.addWidget(content_area, 1)

        self.update_status_label(0)
        self.update_gauge_chart(0)

    def _step_conf(self, delta):
        new_val = round(max(0.05, min(0.95, self.spin_conf.value() + delta)), 2)
        self.spin_conf.setValue(new_val)

    def _on_splitter_moved(self, pos, index):
        if self._sidebar_visible:
            sizes = self.main_splitter.sizes()
            if len(sizes) == 2 and sizes[1] >= 200:
                self._saved_sidebar_width = sizes[1]
        if not self.timer.isActive():
            self._render_last_pixmap()

    def _apply_responsive_layout(self, width, height):
        """Ajusta proporciones de cabecera, logos, márgenes y panel según el tamaño de pantalla."""
        if width < 1080 or height < 650:
            mode = "compact"
        elif width < 1480:
            mode = "medium"
        else:
            mode = "large"

        if mode == self._layout_mode:
            return
        self._layout_mode = mode

        if mode == "compact":
            bar_h = 52
            logo_l_size = QSize(100, 32)
            logo_r_size = QSize(90, 32)
            title_px = 13
            margin = 8
            side_margin = 10
            side_spacing = 8
            sidebar_w = 285
            log_min_h = 80
            toggle_size = QSize(42, 34)
            fs_size = QSize(36, 34)
        elif mode == "medium":
            bar_h = 60
            logo_l_size = QSize(170, 44)
            logo_r_size = QSize(155, 44)
            title_px = 16
            margin = 12
            side_margin = 13
            side_spacing = 9
            sidebar_w = 330
            log_min_h = 105
            toggle_size = QSize(44, 32)
            fs_size = QSize(38, 32)
        else:
            bar_h = 74
            logo_l_size = QSize(240, 56)
            logo_r_size = QSize(220, 56)
            title_px = 19
            margin = 16
            side_margin = 16
            side_spacing = 11
            sidebar_w = 380
            log_min_h = 130
            toggle_size = QSize(46, 34)
            fs_size = QSize(40, 34)

        self.title_bar.setFixedHeight(bar_h)
        self.title_bar_layout.setContentsMargins(margin, 4, margin, 4)
        self.title_label.setStyleSheet(
            f"font-size: {title_px}px; font-weight: 600; color: #1A1A1A; background: transparent;"
        )
        self.toggle_sidebar_btn.setFixedSize(toggle_size)
        self.toggle_fullscreen_btn.setFixedSize(fs_size)

        if self.logo_left_pixmap is not None:
            self.logo_left.setPixmap(
                self.logo_left_pixmap.scaled(logo_l_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        if self.logo_right_pixmap is not None:
            self.logo_right.setPixmap(
                self.logo_right_pixmap.scaled(logo_r_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        self.content_layout.setContentsMargins(margin, margin, margin, margin)
        self.sidebar_layout.setContentsMargins(side_margin, side_margin, side_margin, side_margin)
        self.sidebar_layout.setSpacing(side_spacing)
        self.log_area.setMinimumHeight(log_min_h)
        self._saved_sidebar_width = sidebar_w

        # Repartir espacio inicial entre visor y barra lateral solo si está visible
        if self._sidebar_visible:
            video_w = max(320, width - sidebar_w - (margin * 2) - 10)
            self.main_splitter.setSizes([video_w, sidebar_w])

    # ------------------------------------------------------------------
    # GESTIÓN DE MODELOS BAJO DEMANDA Y PARÁMETROS DE INFERENCIA
    # ------------------------------------------------------------------
    def _resolve_model_path(self, filename):
        """Busca el archivo de pesos en models y luego en el directorio de ejecución."""
        direct_path = MODELS_DIR / filename
        if direct_path.is_file():
            return str(direct_path)
        cwd_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(cwd_path):
            return cwd_path
        raise FileNotFoundError(f"No se encontró '{filename}' en '{MODELS_DIR}'.")

    def select_process_stage(self, stage_key, load_now=True):
        """Selecciona la etapa (Escarmenado, Cardado o Hilado) y carga el modelo bajo demanda."""
        if stage_key not in PROCESS_MODELS:
            return

        self.current_stage_key = stage_key
        for key, btn in self.stage_buttons.items():
            btn.setChecked(key == stage_key)

        cfg = PROCESS_MODELS[stage_key]
        tiled = cfg.get("tiled", False)
        self.standard_params.setVisible(not tiled)
        self.tiled_params.setVisible(tiled)
        self._reset_tiled_summary()
        if not load_now:
            self.model_info_label.setText(f"{cfg['filename']} · {cfg['desc']} (en espera)")
            return

        ok = self._ensure_model_loaded(stage_key)
        if ok and self.test_image_bgr is not None and not self.timer.isActive():
            self._run_test_image_inference()

    def _ensure_model_loaded(self, stage_key=None):
        """Carga bajo demanda el modelo correspondiente y libera el anterior."""
        if stage_key is None:
            stage_key = self.current_stage_key or "escarmenado"

        cfg = PROCESS_MODELS[stage_key]
        if self.model is not None and PROCESS_MODELS[self.loaded_model_key]["filename"] == cfg["filename"]:
            self.loaded_model_key = stage_key
            return True

        if YOLO is None:
            QMessageBox.critical(
                self,
                "Error de Modelo",
                f"No se pudo importar Ultralytics YOLO.\n{YOLO_IMPORT_ERROR}"
            )
            return False

        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            override = os.environ.get("ALPACA_TILING_WEIGHTS") if cfg.get("tiled") else None
            model_path = override or self._resolve_model_path(cfg["filename"])
            self.model_info_label.setText(f"Cargando {os.path.basename(model_path)}…")

            # Liberar modelo previo antes de cargar el nuevo
            if self.model is not None:
                del self.model
                self.model = None
                self.loaded_model_key = None
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if cfg.get("tiled"):
                self.model = TiledImpurityDetector(
                    model_path, tile_batch_size=20 if torch.cuda.is_available() else 4,
                )
            else:
                self.model = YOLO(model_path)
            self.loaded_model_key = stage_key
            self.model_info_label.setText(f"{os.path.basename(model_path)} · {cfg['desc']}")
            print(f"Modelo cargado [{cfg['label']}]: {model_path}")
            return True

        except Exception as e:
            self.model = None
            self.loaded_model_key = None
            self.model_info_label.setText(f"Error al cargar {cfg['filename']}")
            QMessageBox.critical(
                self,
                "Error de Modelo",
                f"No se pudo cargar el modelo '{cfg['filename']}'.\n{e}"
            )
            return False

        finally:
            if was_running:
                if self.model is not None:
                    self.timer.start(33)
                else:
                    self._stop_detection()

    def _on_infer_params_changed(self):
        """Actualiza los controles y vuelve a procesar la imagen de prueba."""
        self.infer_conf = float(self.spin_conf.value())
        self.infer_imgsz = int(self.combo_imgsz.currentData())

        if self.test_image_bgr is not None and not self.timer.isActive():
            self._run_test_image_inference()

    def _active_stage_config(self):
        key = self.loaded_model_key or self.current_stage_key
        return PROCESS_MODELS.get(key, {})

    def _is_segmentation_stage(self):
        return self._active_stage_config().get("task") == "segment"

    def _get_stage_class_names(self, stage_cfg):
        """Devuelve el mapa de nombres de clases aplicando overrides de la etapa si existen."""
        base_names = self.model.names if self.model is not None else {}
        override = stage_cfg.get("class_names_override")
        if not override:
            return base_names

        if isinstance(base_names, dict):
            merged = dict(base_names)
            merged.update(override)
            return merged

        merged = {idx: name for idx, name in enumerate(base_names)}
        merged.update(override)
        return merged

    def _reset_tiled_summary(self):
        if self.tiled_job is not None:
            self.tiled_job[0].cancel()
            self.tiled_job = None
        self.tiled_timer.stop()
        self.last_tiled_detections = []
        self.tiled_summary.setText("Tiles de 640 px · solape de 128 px")

    def _run_tiled_inference(self, frame):
        audit = self.tiled_view.currentIndex() == 1
        mode = "batched_nms" if self.timer.isActive() and not audit else "cross_tile_merge"
        live = self.timer.isActive()
        if self.tiled_job is not None:
            self.tiled_job[0].cancel()
        if not live or self.last_rendered_pixmap is None:
            self.tiled_summary.setText("Procesando tiles…")
            self._display_bgr_frame(frame)
        future = self.tiled_executor.submit(
            self.model.predict_frame, frame, conf_particle=self.spin_particle.value(),
            conf_straw=self.spin_straw.value(), dedup=mode,
        )
        self.tiled_job = (future, frame, audit, live)
        self.tiled_timer.start()

    def _finish_tiled_inference(self):
        if self.tiled_job is None:
            return
        future, frame, audit, live = self.tiled_job
        if not future.done():
            return
        self.tiled_job = None
        self.tiled_timer.stop()
        try:
            detections, metrics = future.result()
        except Exception as exc:
            if live:
                self._stop_detection()
            else:
                self._reset_tiled_summary()
                self.update_status_label(0)
                self.update_gauge_chart(0)
            QMessageBox.warning(self, "Error de inferencia", str(exc))
            return
        self.last_tiled_detections = detections
        count = len(detections)
        particles = sum(d["class_id"] == 0 for d in detections)
        self.tiled_summary.setText(
            f"{len(metrics['tiles'])} tiles · {metrics['total_ms']:.0f} ms\n"
            f"Partículas: {particles} · Paja: {count - particles}"
        )
        self.update_status_label(count)
        self.update_gauge_chart(count)
        self._display_bgr_frame(render_detections(frame, detections, metrics, audit=audit))

    def _run_inference_on_frame(self, frame):
        """
        Ejecuta la inferencia YOLO compartida tanto para video en tiempo real
        como para la verificación temporal con imágenes.
        """
        if self.model is not None and self._active_stage_config().get("tiled"):
            self._run_tiled_inference(frame)
            return
        if self.model is not None:
            results = self.model(
                frame,
                conf=self.infer_conf,
                imgsz=self.infer_imgsz,
                verbose=False
            )

            detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
            self.update_status_label(detection_count)
            self.update_gauge_chart(detection_count)

            stage_cfg = self._active_stage_config()
            class_names = self._get_stage_class_names(stage_cfg)
            annotated_frame = draw_custom_predictions(
                frame.copy(),
                results,
                class_names,
                draw_masks=self._is_segmentation_stage(),
                color_map=stage_cfg.get("colors")
            )

            speed = results[0].speed
            stage_label = stage_cfg.get("label", "Modelo")
            log_msg = (
                f"[{stage_label}] {detection_count} detections "
                f"(conf={self.infer_conf:.2f}, imgsz={self.infer_imgsz}): "
                f"{speed['preprocess']:.1f}ms pre, {speed['inference']:.1f}ms inf, {speed['postprocess']:.1f}ms post"
            )
            print(log_msg)
        else:
            annotated_frame = frame.copy()

        self._display_bgr_frame(annotated_frame)

    def _display_bgr_frame(self, bgr_frame):
        """Convierte un frame BGR de OpenCV a QPixmap y lo muestra escalado en el visor."""
        rgb_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.last_rendered_pixmap = QPixmap.fromImage(qt_image)
        self._render_last_pixmap()

    def _render_last_pixmap(self):
        if self.last_rendered_pixmap is None:
            return
        target = self.video_label.size()
        if target.width() > 10 and target.height() > 10:
            self.video_label.setPixmap(
                self.last_rendered_pixmap.scaled(
                    target,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(self.width(), self.height())
        if not self.timer.isActive():
            self._render_last_pixmap()

    # ------------------------------------------------------------------
    # === MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES ===
    # (Se puede eliminar luego sin afectar la detección en tiempo real)
    # ------------------------------------------------------------------
    def _open_test_image(self):
        """Carga una imagen local para verificar el modelo y parámetros sin cámara."""
        if self.timer.isActive():
            QMessageBox.information(
                self,
                "Prueba con imagen",
                "Detén la cámara en tiempo real antes de cargar una imagen de prueba."
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen de prueba",
            self.base_dir,
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not file_path:
            return

        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("No se pudo decodificar la imagen seleccionada.")

            self.test_image_bgr = img
            self.test_image_path = file_path
            self.btn_clear_test_image.setEnabled(True)
            print(f"Imagen de prueba cargada: {os.path.basename(file_path)} ({img.shape[1]}x{img.shape[0]})")

            self._run_test_image_inference()
        except Exception as e:
            QMessageBox.warning(self, "Error de imagen", f"No se pudo abrir la imagen.\n{e}")

    def _run_test_image_inference(self):
        """Ejecuta inferencia sobre la imagen de prueba cargada usando el modelo seleccionado."""
        if self.test_image_bgr is None:
            return
        if not self._ensure_model_loaded(self.current_stage_key):
            return
        try:
            self._run_inference_on_frame(self.test_image_bgr)
        except Exception as e:
            self._reset_tiled_summary()
            self.update_status_label(0)
            self.update_gauge_chart(0)
            self._display_bgr_frame(self.test_image_bgr)
            QMessageBox.warning(self, "Error de inferencia", str(e))

    def _clear_test_image(self):
        """Limpia la imagen de prueba cargada."""
        self.test_image_bgr = None
        self.test_image_path = None
        self.last_rendered_pixmap = None
        self.btn_clear_test_image.setEnabled(False)
        if not self.timer.isActive():
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("Inicie la detección para ver la cámara o cargue una imagen de prueba")
            self.update_status_label(0)
            self.update_gauge_chart(0)
        self._reset_tiled_summary()
        print("Imagen de prueba limpiada.")
    # === FIN MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES ===

    def _apply_styles(self):
        """Estilo neutro y plano de instrumento industrial, sin saturación y táctil."""
        self.setStyleSheet("""
            * {
                font-family: 'Segoe UI', 'Ubuntu', 'Arial', sans-serif;
                color: #1F1F1F;
            }
            QMainWindow, #contentArea {
                background-color: #EFEFEF;
            }
            QSplitter::handle {
                background-color: #EFEFEF;
            }
            QSplitter::handle:hover {
                background-color: #D6D6D6;
            }
            #titleBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #D6D6D6;
            }
            #logoBox {
                background: transparent;
                color: #666666;
                font-size: 11px;
            }
            #videoLabel {
                background-color: #161616;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                color: #A3A3A3;
                font-size: 14px;
                padding: 12px;
            }
            #sidebarScroll {
                background-color: #FFFFFF;
                border: 1px solid #D6D6D6;
                border-radius: 3px;
            }
            #sidebar {
                background-color: #FFFFFF;
                border: none;
            }
            #toggleSidebarBtn {
                background-color: #9E2A2B;
                border: 1px solid #7D1F20;
                border-radius: 4px;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: 700;
                padding: 0px;
            }
            #toggleSidebarBtn:hover {
                background-color: #B43536;
                border-color: #8A2425;
            }
            #toggleSidebarBtn:pressed {
                background-color: #7D1F20;
            }
            #fullscreenBtn {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                color: #262626;
                font-size: 14px;
                font-weight: 600;
                padding: 0px;
            }
            #fullscreenBtn:hover {
                background-color: #EFEFEF;
                border-color: #B3B3B3;
            }
            #touchStepBtn {
                background-color: #F4F4F4;
                border: 1px solid #C8C8C8;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 700;
                padding: 0px;
            }
            #touchStepBtn:pressed {
                background-color: #DFDFDF;
            }
            #divider {
                color: #E5E5E5;
                background-color: #E5E5E5;
                max-height: 1px;
                border: none;
            }
            #sectionLabel {
                font-size: 12px;
                font-weight: 600;
                color: #262626;
            }
            #mutedLabel {
                font-size: 11px;
                color: #5E5E5E;
            }
            #monoValue {
                font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 11px;
                font-weight: 600;
                color: #262626;
            }
            #monoMuted {
                font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 10px;
                color: #666666;
            }
            #stageBtn {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 6px 4px;
                font-size: 11px;
                font-weight: 500;
                color: #262626;
            }
            #stageBtn:hover {
                background-color: #EFEFEF;
            }
            #stageBtn:checked {
                background-color: #262626;
                color: #FFFFFF;
                border: 1px solid #1A1A1A;
                font-weight: 600;
            }
            #detectButton {
                background-color: #262626;
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 12px;
                border: 1px solid #1A1A1A;
                border-radius: 3px;
            }
            #detectButton:hover {
                background-color: #383838;
            }
            #detectButton:pressed {
                background-color: #171717;
            }
            #detectButton[running="true"] {
                background-color: #F4F4F4;
                color: #8A1F1D;
                border: 1px solid #C98A88;
            }
            #detectButton[running="true"]:hover {
                background-color: #FBECEB;
            }
            #statusLabel {
                font-size: 12px;
                font-weight: 600;
                padding: 8px 10px;
                border-radius: 3px;
            }
            #statusLabel[status="clean"] {
                background-color: #EEF5F0;
                color: #235334;
                border: 1px solid #C6DEC9;
                border-left: 4px solid #3A7D54;
            }
            #statusLabel[status="warning"] {
                background-color: #FBF5E8;
                color: #7A520B;
                border: 1px solid #E8D2A6;
                border-left: 4px solid #C28522;
            }
            #statusLabel[status="danger"] {
                background-color: #FAEEEE;
                color: #7D2322;
                border: 1px solid #E3C0BF;
                border-left: 4px solid #B03A38;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 4px 5px;
                font-size: 11px;
                color: #1F1F1F;
            }
            QPushButton {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 6px 8px;
                font-size: 11px;
                font-weight: 500;
                color: #1F1F1F;
            }
            QPushButton:hover {
                background-color: #EFEFEF;
                border-color: #B3B3B3;
            }
            QPushButton:pressed {
                background-color: #E5E5E5;
            }
            QPushButton:disabled {
                color: #A0A0A0;
                background-color: #F5F5F5;
                border-color: #E0E0E0;
            }
            #motorActionBtn {
                border-left: 3px solid #3A7D54;
            }
            #motorStopBtn {
                border-left: 3px solid #B03A38;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #D9D9D9;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #595959;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #262626;
                border: 1px solid #1A1A1A;
                width: 24px;
                height: 24px;
                margin: -8px 0;
                border-radius: 12px;
            }
            QTextEdit {
                background-color: #F7F7F7;
                border: 1px solid #D9D9D9;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 10px;
                color: #333333;
            }
        """)

    def _set_running_visual_state(self, running):
        self.detect_button.setProperty("running", "true" if running else "false")
        self.detect_button.style().unpolish(self.detect_button)
        self.detect_button.style().polish(self.detect_button)

    def log(self, message):
        """Añade un mensaje al área de texto y hace scroll automático al final."""
        self.log_area.append(str(message))
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    # ------------------------------------------------------------------
    # LÓGICA DE DETECCIÓN EN TIEMPO REAL (sin alterar su flujo original)
    # ------------------------------------------------------------------
    def _stop_detection(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.detect_button.setText("Iniciar detección")
        self._set_running_visual_state(False)
        self.last_rendered_pixmap = None
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Detección detenida.")
        self._reset_tiled_summary()

    def start_detection(self):
        """Alterna el estado de la detección (Iniciar/Detener)."""
        if self.timer.isActive():
            self._stop_detection()
            return

        # Cargar el modelo de la etapa seleccionada bajo demanda antes de abrir la cámara
        if not self._ensure_model_loaded(self.current_stage_key):
            return

        # Limpiar cualquier imagen temporal cargada antes de entrar en tiempo real
        if self.test_image_bgr is not None:
            self.test_image_bgr = None
            self.test_image_path = None
            self.btn_clear_test_image.setEnabled(False)

        try:
            self.cap = create_camera()
            self.cap.open()
        except Exception as e:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            print(f"Error al abrir la cámara: {e}")
            QMessageBox.critical(
                self,
                "Error de cámara",
                f"No se pudo abrir la cámara.\n{e}"
            )
            self._set_running_visual_state(False)
            return

        self.detect_button.setText("Detener detección")
        self._set_running_visual_state(True)
        self.timer.start(33)
        print("Iniciando detección...")

    def update_status_label(self, count):
        """
        Actualiza la etiqueta de estado basándose en umbrales de conteo de impurezas.
        Utiliza propiedades dinámicas de Qt para cambiar el estilo (colores).
        """
        if count < 5:
            status = "clean"
            text = "FIBRA LIMPIA"
        elif count <= 20:
            status = "warning"
            text = "FIBRA CON IMPUREZAS"
        else:
            status = "danger"
            text = "FIBRA SUCIA"

        self.status_label.setText(text)
        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def update_gauge_chart(self, count):
        """Renderiza una barra horizontal limpia y sobria para el nivel de impurezas."""
        max_count = 60
        clamped_count = max(0, min(count, max_count))
        count_norm = clamped_count / max_count

        if hasattr(self, "gauge_value_label"):
            self.gauge_value_label.setText(f"{count} / {max_count}")

        self.gauge_figure.clear()
        ax = self.gauge_figure.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        cmap = mcolors.LinearSegmentedColormap.from_list("grad", ["#3A7D54", "#C28522", "#B03A38"])
        bar_color = cmap(count_norm)

        ax.barh([0], [max_count], color="#EAEAEA", height=0.62, edgecolor="#D4D4D4", linewidth=0.7)
        if clamped_count > 0:
            ax.barh([0], [clamped_count], color=bar_color, height=0.62, edgecolor="none")

        ax.set_xlim(0, max_count)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.gauge_figure.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.08)
        self.gauge_canvas.draw()

    def update_frame(self):
        """Ciclo principal: Captura frame -> Inferencia YOLO -> Actualización UI."""
        if self.tiled_job is not None or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._stop_detection()
            QMessageBox.warning(self, "Advertencia", "Se perdió la conexión con la cámara.")
            return

        try:
            self._run_inference_on_frame(frame)
        except Exception as exc:
            self._stop_detection()
            QMessageBox.warning(self, "Error de inferencia", str(exc))

    def closeEvent(self, event):
        """Limpieza de recursos al cerrar la ventana."""
        print("Cerrando aplicación...")
        self.timer.stop()
        self._reset_tiled_summary()
        self.tiled_executor.shutdown(wait=False)
        if self.cap is not None:
            self.cap.release()
            print("Cámara liberada.")

        try:
            self.motor_serial_timer.stop()
            if self.serial_port is not None:
                self._send_motor_command("STOP")
                self.serial_port.close()
                self.serial_port = None
        except Exception:
            pass

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTabletEvents, True)
    QApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())