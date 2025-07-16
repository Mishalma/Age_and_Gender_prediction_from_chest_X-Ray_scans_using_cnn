# 🩻 Age and Gender Prediction from Chest X-ray Scans using CNN

This project builds a Convolutional Neural Network (CNN) to predict **age** and **gender** from chest X-ray scans. The data is preprocessed using grayscale conversion, inversion, resizing, and normalization. A custom CNN model is trained separately for binary gender classification and age regression.

---

## 📁 Dataset

- **Source**: [KaggleHub - SPR X-ray Age and Gender Dataset](https://www.kaggle.com/datasets/felipekitamura/spr-x-ray-age-and-gender-dataset)
- **Structure**:


/dataset/
├── train/
├── train_age.csv
└── train_gender.csv

- **Labels**:
- `train_age.csv` → age (in years)
- `train_gender.csv` → gender (0: female, 1: male)

---

## ⚙️ Preprocessing Pipeline

1. Convert to grayscale using OpenCV
2. Invert image (negative): `255 - pixel`
3. Resize to `128x128`
4. Normalize pixel values to `[0, 1]`
5. Merge image, age, and gender into a unified DataFrame

---

## 🧠 Model Architecture

A custom CNN built using TensorFlow/Keras:

### 🔹 Gender Prediction (Binary Classification)

- **Input**: `128x128x1` (grayscale image)
- **Blocks**: Repeated Conv2D → MaxPool → BatchNorm → ReLU
- **Output Layer**: `Dense(1, activation='sigmoid')`
- **Loss**: `binary_crossentropy`
- **Optimizer**: `Adam`

### 🔹 Age Prediction (Regression)

- Same architecture as above
- Final Layer: `Dense(1)` (linear activation)
- **Loss**: To be implemented with `mean_squared_error` or `mean_absolute_error`

---

## 🏋️‍♂️ Training the Model

### ✅ Gender Classification

```python
model_gender.fit(
  X, y,
  steps_per_epoch=1000,
  validation_split=0.2,
  epochs=5
)

### 🚫 Age Prediction
Defined but training not executed yet. Update label to Age and change output activation & loss.

### 💾 Model Saving
model_gender.save('model_gender.h5')

### 📊 Evaluation
Evaluation metrics can be added using:
model.evaluate(X_test, y_test)

### 🚀 To-Do / Future Work
✅ Train gender prediction model

⏳ Finish and train age regression model

📈 Add visualizations (Grad-CAM, heatmaps)

♻️ Use transfer learning (e.g., ResNet, Inception)

🧪 Add robust evaluation + test-time augmentation

📄 License
This project is licensed under the MIT License. See LICENSE for details.
