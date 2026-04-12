# 🧠 Customer Churn Prediction — ANN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)
![License](https://img.shields.io/badge/License-GPL--3.0-green)

**[🚀 Live Demo](https://q99dnzatazcgpzbdavtuma.streamlit.app/)**

---

## Overview

End-to-end deep learning solution to predict bank customer churn. Given customer demographics and account data, the model outputs a **churn probability (0–1)** and a yes/no verdict — enabling retention teams to intervene proactively.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Model | TensorFlow / Keras (ANN) |
| Preprocessing | scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler) |
| App | Streamlit |
| Experiment Tracking | TensorBoard |

---

## Model Architecture

- **Input:** 11 features (after encoding)
- **Hidden Layers:** 64 → 32 neurons, ReLU activation
- **Output:** Sigmoid (binary churn probability)
- **Loss:** Binary Cross-Entropy | **Optimizer:** Adam
- **Callbacks:** EarlyStopping, TensorBoard

---

## Dataset

10,000 bank customer records. Features include credit score, geography, gender, age, balance, tenure, number of products, and activity status. Target: `Exited` (1 = churned).

---

## Key Skills Demonstrated

- Designing and training a multi-layer ANN for binary classification
- Full preprocessing pipeline: label encoding, one-hot encoding, standard scaling
- Saving/loading Keras models (`.h5`) and scikit-learn objects (`.pkl`) to prevent training-serving skew
- Building and deploying an interactive ML app with Streamlit
- Bonus: ANN adapted for **regression** (salary prediction) in `salaryregression.ipynb`

---

## Repo Structure

```
├── Churn_Modelling.csv        # Dataset
├── experiments.ipynb          # EDA → training → evaluation
├── prediction.ipynb           # Inference on new data
├── salaryregression.ipynb     # Bonus: regression with ANN
├── app.py                     # Streamlit app
├── model.h5                   # Trained model
├── scaler.pkl / *.pkl         # Preprocessing artifacts
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/kuldeepsingh6/ANN-churn-.git
cd ANN-churn-
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Kuldeep Singh** · [GitHub](https://github.com/kuldeepsingh6)
