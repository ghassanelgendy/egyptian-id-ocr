# setup.py
from setuptools import setup

setup(
    name="ocr_egyptian_ID",
    version="1.0.0",
    description="Dynamic OCR preprocessing, name correction, and decoding pipeline for Egyptian ID Cards",
    author="Ghassan",
    py_modules=["utils"],
    install_requires=[
        "numpy",
        "opencv-python",
        "easyocr",
        "ultralytics"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
