# 📩 Message Intelligence System

## 📌 Project Overview

The **Message Intelligence System** is a Machine Learning project that automatically classifies digital messages into:

* **0 → Legitimate Message**
* **1 → Spam Message**

This project applies concepts of **Probability Theory**, **Distance-Based Learning**, **Margin-Based Learning**, and **Probabilistic Classification** to compare different machine learning algorithms for spam detection.

---

## 🎯 Objective

The main objective of this project is to design and implement a message classification system that can identify whether an incoming message is **Spam** or **Legitimate**.

The project compares the performance of:

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes Classifier

---

## 📂 Dataset Description

The dataset contains message-related features such as:

* Message Length
* Number of Special Characters
* Number of URLs
* Keyword Frequency Score
* Sender Activity Score
* Time-Based Features
* Target Variable: `Spam_Label`

### Target Labels

| Label | Meaning            |
| ----- | ------------------ |
| 0     | Legitimate Message |
| 1     | Spam Message       |

---

## 🛠 Technologies Used

* Python 3.x
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📦 Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 📁 Project Structure

```
Message-Intelligence-System/
│
├── Message_Intelligence_Dataset(2).csv
├── Message_Intelligence_System.ipynb
├── README.md
└── screenshots/
```

---

## 🚀 Project Workflow

### Step 1: Import Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Step 2: Load Dataset

* Read CSV file
* Display dataset information
* Check data types and missing values

### Step 3: Data Preprocessing

* Identify input features and target variable
* Split dataset into training and testing sets
* Standardize numerical features

### Step 4: Model Building

#### 1. K-Nearest Neighbors (KNN)

* Distance-based classifier
* Experiment with different values of K

#### 2. Support Vector Machine (SVM)

* Linear Kernel
* RBF Kernel
* Margin maximization technique

#### 3. Naive Bayes Classifier

* Probabilistic classifier
* Uses Bayes' Theorem

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📈 Results

The performance of all classifiers is compared using:

* Accuracy Comparison Graph
* Classification Report
* Confusion Matrix
* Model Performance Table

---

## 🧠 Theory Concepts Used

### Conditional Probability

```
P(A|B) = P(A ∩ B) / P(B)
```

### Bayes' Theorem

```
P(A|B) = (P(B|A) × P(A)) / P(B)
```

### Machine Learning Algorithms

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes Classifier

---

## ✅ Conclusion

This project demonstrates how machine learning and probability concepts can be used to detect spam messages automatically.

* KNN is simple and easy to understand.
* SVM generally provides high classification accuracy.
* Naive Bayes is fast and computationally efficient.

The Message Intelligence System can be used in email filtering, SMS spam detection, and communication security applications.

---

### ⭐ If you like this project, give it a star on GitHub.
