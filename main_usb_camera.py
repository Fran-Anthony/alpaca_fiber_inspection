# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "ultralytics>=8.4.138",
#     "torch>=2.0.0",
#     "torchvision>=0.15.1",
#     "opencv-python>=5.0.0.93",
#     "numpy>=1.24.4",
#     "PyQt5>=5.14.1",
#     "pyserial>=3.5",
#     "Pillow>=9.1",
# ]
# ///
from alpaca_inspection.inspection_gui import MainWindow, main


if __name__ == "__main__":
    main()
