# 🧠 Customer Churn Prediction using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn)
![License](https://img.shields.io/badge/License-GPL--3.0-green)

---

## 📌 Project Overview

Customer churn — when a customer stops doing business with a company — is one of the most critical challenges in the banking and financial services industry. Retaining an existing customer is far more cost-effective than acquiring a new one.

This project builds an end-to-end **deep learning solution** to predict whether a bank customer is likely to leave (churn) based on their demographic and account information. The model is built using an **Artificial Neural Network (ANN)** with TensorFlow/Keras, trained on real-world-style banking data, and deployed as an **interactive web application** using Streamlit.

> ✅ This is a complete ML project — from raw data preprocessing, to model training, to a live interactive app.

---

## 🎯 Problem Statement

A bank wants to proactively identify customers who are at risk of leaving so that the retention team can reach out with targeted offers. Given a set of customer attributes (credit score, age, balance, activity status, etc.), the model predicts:

- **Will this customer churn? (Yes/No)**
- **What is the probability of churn? (0.0 – 1.0)**

---

## 🗂️ Repository Structure

```
ANN-churn-/
│
├── Churn_Modelling.csv          # Dataset (10,000 bank customer records)
├── experiments.ipynb            # Full model training pipeline (EDA → Preprocessing → ANN → Evaluation)
├── prediction.ipynb             # Notebook for making predictions on new/test data
├── salaryregression.ipynb       # Bonus: ANN-based regression to predict estimated salary
│
├── app.py                       # Streamlit web application for live predictions
│
├── model.h5                     # Saved trained ANN model (TensorFlow/Keras)
├── scaler.pkl                   # Fitted StandardScaler (for feature normalization)
├── label_encoder_gender.pkl     # Fitted LabelEncoder (for Gender column)
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder (for Geography column)
│
├── requirements.txt             # All project dependencies
└── README.md                    # Project documentation (you are here)
```

---

## 📊 Dataset Description

**File:** `Churn_Modelling.csv`  
**Source:** Public banking churn dataset  
**Records:** 10,000 customers  
**Target Variable:** `Exited` (1 = Churned, 0 = Stayed)

| Feature | Description | Type |
|---|---|---|
| `CreditScore` | Customer's credit score | Numerical |
| `Geography` | Country of the customer (France, Germany, Spain) | Categorical |
| `Gender` | Male / Female | Categorical |
| `Age` | Age of the customer | Numerical |
| `Tenure` | Number of years as a bank customer | Numerical |
| `Balance` | Account balance | Numerical |
| `NumOfProducts` | Number of bank products held | Numerical |
| `HasCrCard` | Whether the customer has a credit card (1/0) | Binary |
| `IsActiveMember` | Whether the customer is an active member (1/0) | Binary |
| `EstimatedSalary` | Estimated annual salary | Numerical |
| `Exited` *(Target)* | Whether the customer churned (1 = Yes, 0 = No) | Binary |

> **Note:** Columns like `RowNumber`, `CustomerId`, and `Surname` were dropped as they carry no predictive signal.

---

## 🏗️ Project Architecture & Workflow

```
Raw CSV Data
     │
     ▼
Data Preprocessing
  ├── Drop irrelevant columns (RowNumber, CustomerId, Surname)
  ├── Label Encode → Gender (Male=1, Female=0)
  ├── One-Hot Encode → Geography (France, Germany, Spain)
  └── Standard Scale → All numerical features
     │
     ▼
ANN Model (TensorFlow / Keras)
  ├── Input Layer  → 11 features
  ├── Hidden Layer 1 → 64 neurons, ReLU activation
  ├── Hidden Layer 2 → 32 neurons, ReLU activation
  └── Output Layer → 1 neuron, Sigmoid activation (binary probability)
     │
     ▼
Training
  ├── Loss: Binary Cross-Entropy
  ├── Optimizer: Adam
  ├── Metric: Accuracy
  └── Callbacks: EarlyStopping, TensorBoard logging
     │
     ▼
Saved Artifacts
  ├── model.h5
  ├── scaler.pkl
  ├── label_encoder_gender.pkl
  └── onehot_encoder_geo.pkl
     │
     ▼
Streamlit Web App (app.py)
  └── Real-time prediction with user inputs
```

---

## 🧪 Model Details

| Parameter | Value |
|---|---|
| Architecture | Artificial Neural Network (ANN) |
| Framework | TensorFlow 2.15.0 / Keras |
| Input Features | 11 (after encoding) |
| Hidden Layers | 2 (64 → 32 neurons) |
| Activation (Hidden) | ReLU |
| Activation (Output) | Sigmoid |
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |
| Callbacks | EarlyStopping, TensorBoard |
| Output | Churn probability (0.0 – 1.0) |

**Decision Rule:** If predicted probability > 0.5 → Customer is likely to churn.

---

## 🔧 Feature Engineering & Preprocessing

Three preprocessing steps were applied and saved as `.pkl` files so they can be reused consistently during inference:

1. **Label Encoding** (`label_encoder_gender.pkl`) — Converts `Gender` (text) → numeric (0/1)
2. **One-Hot Encoding** (`onehot_encoder_geo.pkl`) — Converts `Geography` → binary columns (e.g., `Geography_France`, `Geography_Germany`, `Geography_Spain`)
3. **Standard Scaling** (`scaler.pkl`) — Normalizes all numerical features to zero mean and unit variance, which is critical for neural network convergence

All encoders/scalers were fitted **only on training data** and applied on test/inference data to prevent data leakage.

---

## 📓 Notebooks Explained

### `experiments.ipynb` — Core Training Notebook
This is the main notebook where everything happens:
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Encoding categorical variables
- Splitting into train/test sets
- Building and compiling the ANN
- Training with early stopping
- Evaluating on the test set (accuracy, loss curves)
- Saving the model and preprocessing artifacts

### `prediction.ipynb` — Inference Notebook
Demonstrates how to load the saved model and artifacts to make predictions on new, unseen customer data — simulating a real-world deployment scenario.

### `salaryregression.ipynb` — Bonus Regression Task
An extension of the project where the same ANN architecture is adapted for **regression** to predict a customer's `EstimatedSalary`. This demonstrates the versatility of neural networks for both classification and regression tasks.

---

## 🖥️ Streamlit Web Application

The `app.py` file provides a fully interactive web UI for making real-time predictions.

**How it works:**
1. User inputs customer details via dropdowns, sliders, and number fields
2. Inputs are preprocessed using the same saved encoders and scaler
3. The preprocessed input is fed to the loaded ANN model
4. The app displays the **churn probability** and a plain-English verdict

**Input fields in the app:**

| Input | Type | Range/Options |
|---|---|---|
| Geography | Dropdown | France, Germany, Spain |
| Gender | Dropdown | Male, Female |
| Age | Slider | 18 – 92 |
| Balance | Number Input | Any |
| Credit Score | Number Input | Any |
| Estimated Salary | Number Input | Any |
| Tenure | Slider | 0 – 10 years |
| Number of Products | Slider | 1 – 4 |
| Has Credit Card | Dropdown | 0 (No), 1 (Yes) |
| Is Active Member | Dropdown | 0 (No), 1 (Yes) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or above
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kuldeepsingh6/ANN-churn-.git
cd ANN-churn-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📦 Dependencies

```
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 💡 Key Technical Skills Demonstrated

- **Deep Learning** — Designing, training, and evaluating a multi-layer ANN for binary classification
- **Feature Engineering** — Label encoding, one-hot encoding, and standard scaling; handling mixed data types
- **Model Persistence** — Saving and loading Keras models (`.h5`) and scikit-learn objects (`.pkl`) with pickle
- **Inference Pipeline** — Reproducing the exact preprocessing pipeline at inference time to avoid training-serving skew
- **Web Deployment** — Building an interactive ML web app with Streamlit
- **Experiment Tracking** — TensorBoard integration for visualizing training metrics
- **Bonus Regression** — Adapting the same architecture for a regression task (salary prediction)

---

## 📈 Business Impact

| Metric | Value |
|---|---|
| Domain | Banking / Financial Services |
| Problem Type | Binary Classification |
| Use Case | Customer Retention & Churn Prevention |
| Decision Support | Enables proactive outreach to at-risk customers |

Predicting churn even a few weeks early allows retention teams to intervene with personalized offers, potentially saving thousands of dollars per customer in acquisition costs.

---

## 🔮 Potential Improvements

- Add **SHAP / LIME explainability** to show why the model predicts churn for a specific customer
- Experiment with **class imbalance handling** (SMOTE, class weights) since churn datasets are typically imbalanced
- Compare ANN performance against classical ML baselines (Random Forest, XGBoost)
- Deploy on **Streamlit Cloud** or **Hugging Face Spaces** for public access
- Add **model retraining pipeline** with new data via CI/CD

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Kuldeep Singh**  
📧 [GitHub Profile](https://github.com/kuldeepsingh6)

---

> ⭐ If you found this project helpful or impressive, consider starring the repository!
