
# 🛰️ Sonar Rock vs Mine Detector

This fun and interactive AI web app detects whether a sonar signal represents a **Rock (🪨)** or a **Mine (💣)** using machine learning. It's built using **Streamlit** and trained on the UCI Sonar Dataset using **Logistic Regression**.

---

## 📌 Problem Statement

The aim is to classify sonar signals as either **Rock** or **Mine**. This is crucial in naval mine detection, where misclassification could lead to:
- False alarms or missed threats
- Ship damage
- Threats to human lives

---

## 💡 Solution Approach

We use a **Logistic Regression** classifier on the UCI Sonar Dataset to make binary predictions based on 60 frequency-based features extracted from sonar signals.

---

## 📊 Dataset

- Source: UCI Machine Learning Repository  
- Each instance consists of **60 numeric features** ranging from 0 to 1, representing sonar energy in different frequency bands.
- Label (`60th column`):  
  - `R` = Rock  
  - `M` = Mine

---

## 🧰 Tech Stack

| Component        | Tool/Library     |
|------------------|------------------|
| Programming      | Python           |
| Web Framework    | Streamlit        |
| ML Library       | Scikit-learn     |
| Data Handling    | Pandas, NumPy    |
| Visualization    | Streamlit Widgets, Images |

---

## 🚀 How to Run the Project

### 🔧 Prerequisites

Install required libraries:

```bash
pip install streamlit scikit-learn pandas numpy
````

### 🏃 Run the App

```bash
streamlit run tpo.py
```

This will launch the app in your browser.

---

## 📌 Features

* Interactive input with **60 sliders** to simulate sonar signal readings
* Real-time predictions and visual feedback (rock/mine image + text)
* Data exploration with expandable dataset summary
* Sidebar showing model accuracy metrics (training and test)

---

## ✅ Model Performance

| Metric            | Score (Sample) |
| ----------------- | -------------- |
| Training Accuracy | \~83%          |
| Test Accuracy     | \~76%          |

*Performance may vary slightly due to random train-test splits.*

---

## 🧠 Queries & Insights

### 1. **How to Improve Accuracy?**

* Try other algorithms like **SVM**, **Random Forest**, or **XGBoost**
* Use **feature scaling**, **dimensionality reduction (PCA)**, or **feature selection**
* Tune hyperparameters using **GridSearchCV**

### 2. **What Features Are Most Relevant?**

* Use `model.coef_` or feature importance techniques (e.g., SHAP, permutation importance)
* Some frequency bands are more discriminative for rocks vs. mines

### 3. **Handling Class Imbalance**

* Check if there's imbalance using `value_counts()`
* Techniques:

  * Use `stratify` in `train_test_split` (already done)
  * Apply **SMOTE** or **undersampling**
  * Use metrics like **ROC AUC**, **F1-score**

---

## 🖼️ Screenshots

| Input Interface                                                                                                                  
| -------------------------------------------                                                                                     
| ![Screenshot 2025-06-14 151148](https://github.com/user-attachments/assets/db19b7df-83a5-47c6-bd0a-a203e0514d95)                 
| ![Screenshot 2025-06-14 151210](https://github.com/user-attachments/assets/f312343c-cbe5-4c8e-a7eb-57bb8aff4f87)

| Prediction Example                       
| ---------------------------------------- 
| ![Screenshot 2025-06-14 151304](https://github.com/user-attachments/assets/77f31791-0122-4958-ae3f-7def8a18e7fc)
| ![Screenshot 2025-06-14 152409](https://github.com/user-attachments/assets/36bcf215-af7d-4b4d-8607-e87eece93013)


---

## 📂 File Structure

```
📁 sonar-rock-mine-detector/
├── tpo.py        # Main Streamlit app
├── sonar data.csv     # Sonar dataset
├── README.md                  # This file
```

---

## ❤️ Acknowledgements

* UCI Machine Learning Repository for the dataset
* Streamlit and Scikit-learn teams for awesome open-source tools

---

## 🔒 License

This project is open source and available under the [MIT License](LICENSE).

---
