"""Rutas del proyecto independientes del directorio de ejecución."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"
RUNTIME_DIR = PROJECT_DIR / "runtime"
