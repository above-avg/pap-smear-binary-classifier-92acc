# 🧬 Pap Smear Binary Classification using ResNet18

Deep learning–based binary classification of cervical cell images (Pap Smear) using **Transfer Learning with ResNet18**, trained and evaluated in **PyTorch**.

This project compares multiple preprocessing strategies and visualizes model attention using **Grad-CAM** for interpretability.

---

## 🚀 Project Overview

This project classifies Pap smear images into:

- **Normal**
- **Abnormal**

The goal is to build a robust binary classifier capable of generalizing across raw and denoised datasets while maintaining high validation accuracy.

---

## 📊 Dataset

- Dataset: **Herlev Pap Smear Dataset (Binary Version)**
- Classes:
  - `normal`
  - `abnormal`
- Total samples:
  - Train: 1467
  - Validation: 367
- Image size: `224 × 224`
- Split: 80% Train / 20% Validation

Dataset structure:
```
dataset/
│
├── normal/
├── abnormal/
```

---

## 🧠 Model Architecture

Base model:
- **ResNet18 (ImageNet pretrained)**

Modified classification head:
```python
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 2)
)
```

Loss Function:
- CrossEntropyLoss

Optimizer:
- Adam (lr = 1e-4)

Epochs:
- 15

---

## 🔄 Data Augmentation

Applied transformations:

- Resize (224x224)
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter
- Gaussian Blur
- Random Resized Crop
- Normalization (ImageNet stats)

These augmentations help improve generalization on medical images.

---

## 📈 Training Results

Final Training Accuracy: **96.25%**  
Final Validation Accuracy: **92.10%**

Example training log:

```
Epoch [15/15] | Train Loss: 0.0989 | Train Acc: 96.25% | Val Acc: 92.10%
```

---

## 🧪 Model Comparison

Three trained variants were evaluated:

| Model | Accuracy | F1 Score |
|--------|----------|----------|
| Denoised (-d) | 92.37% | 0.895 |
| Raw BMP (15 epoch) | 92.37% | 0.890 |
| Raw BMP | 91.28% | 0.881 |

### Example Confusion Matrix (Denoised)

```
[[265  10]
 [ 18  74]]
```

---

## 🔍 Model Interpretability (Grad-CAM)

Grad-CAM was used to visualize which regions influence predictions.

Highlights:

- Focused activations on nucleus region for abnormal cells
- Attention maps consistent across different preprocessing models
- Helps validate medical relevance of learned features

Grad-CAM target layer:
```
model.layer4[-1]
```

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- OpenCV
- Grad-CAM
- Google Colab (GPU)

---

## 📂 Project Workflow

1. Mount Google Drive
2. Load and copy dataset locally
3. Apply augmentation
4. Train ResNet18
5. Save trained models
6. Evaluate multiple models
7. Compare metrics
8. Visualize Grad-CAM results

---

## ⚙️ Installation

Install dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib opencv-python grad-cam
```

Run in:
- Google Colab (recommended with GPU)
- Local machine with CUDA support

---

## 💾 Model Saving

Models are saved to:

```
/content/drive/MyDrive/pap-smear/models/
```

Example:
```
resnet18_binary_raw_generalized_15epoch.pth
```

---

## 🎯 Key Takeaways

- Transfer learning works effectively for medical imaging.
- Strong augmentation improves robustness.
- Grad-CAM adds essential interpretability.
- Denoising does not significantly outperform raw images in this setup.
- Validation accuracy consistently above 92%.

---

## 🔮 Future Improvements

- Add EfficientNet / DenseNet comparison
- Implement K-fold cross-validation
- Add ROC-AUC curves
- Deploy as a web app
- Add multi-class classification (full Herlev dataset)
- Hyperparameter tuning (LR scheduling, weight decay)

---

## ⚠️ Disclaimer

This project is for research and educational purposes only.  
Not intended for clinical or diagnostic use.

---

## 📜 License

Open-source for academic and research purposes.
