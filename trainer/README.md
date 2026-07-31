# Egyptian ID OCR Recognition Fine-Tuning Trainer 🧠

This folder contains a complete, self-contained PyTorch machine learning pipeline to fine-tune a custom character-recognition model on synthetic Egyptian ID names.

It resolves font/ink-level misreadings (such as `ج` vs `ي`) dynamically by training the network to focus strictly on letter strokes and ignore background security watermarks.

---

## 📂 Folder Structure

```
trainer/
├── fonts/                       # Bundled TrueType Arabic fonts
├── generate_synthetic_data.py   # Renders synthetic training name crops
├── train_recognition.py         # PyTorch CRNN model + CTC Loss trainer
├── predict.py                   # Custom image crop prediction script
└── README.md                    # This documentation guide
```

---

## 🚀 How to Run the Pipeline

### **Step 1: Generate the Synthetic Dataset**
Renders high-fidelity name crops using system-like Arabic fonts overlaid with random diagonal lines, gradients, and salt-and-pepper noise to simulate ID card backgrounds.

```bash
python3 generate_synthetic_data.py
```
*Creates `dataset/images/` and `dataset/labels.txt` containing 260+ samples.*

### **Step 2: Start Fine-Tuning**
Trains a Convolutional Recurrent Neural Network (CRNN) using CTC Loss to recognize the sequences.

```bash
python3 train_recognition.py
```
*Saves the trained weights to `custom_arabic_recognition.pth`.*

### **Step 3: Run Inference**
Run prediction on any cropped image field (like `firstName_raw.jpg`):

```bash
python3 predict.py path/to/crop_image.jpg
```

---

## 🔬 Model Architecture (CRNN)

1. **Feature Extractor (CNN):** 7-layer ResNet-like convolutions extracting dynamic shape matrices.
2. **Sequential Modeler (RNN):** 2-layer Bidirectional LSTM capturing character dependencies.
3. **CTC Loss Decoder:** Connectionist Temporal Classification resolving unaligned text frames.
