# 🧠 Handwritten Digit Classification using CNN

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.

The objective is to build a deep learning model capable of accurately recognizing digit images while addressing potential class imbalance using oversampling techniques.

This project demonstrates:
- Image preprocessing
- CNN model building
- Handling class imbalance
- Model evaluation and visualization

---

## 📊 Dataset Information

Dataset: MNIST Handwritten Digits

- 28 × 28 grayscale images
- 784 pixel features (flattened)
- 10 output classes (Digits 0–9)
- Multi-class classification problem

### Dataset Structure

Each row contains:
- `label` → Target digit (0–9)
- `pixel0` to `pixel783` → Pixel intensity values (0–255)

Total Features: 784  
Target Variable: label  

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## 🧪 Data Preprocessing Steps

- Checked class distribution
- Normalized pixel values (scaled between 0–1)
- Reshaped data into (28, 28, 1) format
- Applied oversampling to handle class imbalance
- Train-test split

---

## 📈 Why Oversampling Was Used?

During preprocessing, class imbalance was observed in the dataset subset used for training.

### Issues with Imbalanced Data:
- Model becomes biased toward majority classes
- Poor recall for minority classes
- Reduced generalization ability
- Lower F1-score for underrepresented digits

### Oversampling Solution:

Oversampling was applied to:
- Increase minority class samples
- Balance class distribution
- Improve fairness in predictions
- Enhance recall and overall performance

This ensured the CNN learned all digit classes equally.

---

## 📸 Screenshots

### <img width="626" height="490" alt="Screenshot 2026-02-23 233958" src="https://github.com/user-attachments/assets/b156081b-4dc6-4f97-a64e-adb18a3f23fe" />


## 🧠 Model Architecture

The CNN model consists of:
- Conv2D Layer
- ReLU Activation
- MaxPooling Layer
- Dropout Layer
- Flatten Layer
- Dense Layer
- Softmax Output Layer

---

## 📊 Model Evaluation

Evaluation metrics used:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Training vs Validation Accuracy Curve
- Training vs Validation Loss Curve

---

<img width="593" height="611" alt="Screenshot 2026-02-23 234008" src="https://github.com/user-attachments/assets/ba13a35a-554e-48d8-adeb-57e3fe898413" />

## 🚀 Results

- High classification accuracy
- Balanced performance across all digit classes
- Improved recall for minority classes after oversampling
- Reduced prediction bias

---

## 📂 Project Structure

```
├── CNN.ipynb
├── MNIST_Train.csv
├── screenshots/
│   ├── before_oversampling.png
│   └── after_oversampling.png
└── README.md
```

---

## 🎯 Future Improvements

- Hyperparameter tuning
- Batch normalization
- Early stopping
- Data augmentation
- Transfer learning

---

## 👨‍💻 Author

Aditya Uttekar  
Aspiring Data Scientist | Machine Learning Enthusiast
