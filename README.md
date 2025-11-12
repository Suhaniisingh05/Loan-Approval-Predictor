# 🏦 Automated Loan Eligibility Classifier

## 🚀 Project Overview

This project implements an **End-to-End Machine Learning Pipeline** designed to predict the likelihood of loan approval based on applicant data. The solution includes a robust classification model deployed as an interactive web application using **Streamlit**.

This solution demonstrates core skills in **Data Science, Predictive Modeling, and Model Deployment (MLOps)**.

***

## ✨ Key Features & Technologies

| Category | Skills / Technologies Used | Description |
| :--- | :--- | :--- |
| **ML Model** | **Binary Classification**, Logistic Regression, Scikit-learn, Model Persistence (`pickle`). | Model predicts a binary outcome (Approved/Rejected) with **78.86% accuracy**. |
| **Data Handling** | **Python**, Pandas, NumPy, **Feature Engineering**, Imputation (Mode/Mean), One-Hot Encoding. | Data cleaning and transformation to prepare mixed-type financial data for modeling. |
| **Deployment** | **Streamlit**, **Git & GitHub**. | Model deployed as a user-friendly, interactive web application for real-time risk assessment. |

***

## 💻 Technical Setup and Execution

### 1. Local Setup Guide

1.  **Clone the repository and navigate to the directory:**
    ```bash
    git clone [YOUR GITHUB REPO LINK]
    cd Loan_Approval_App
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # For Windows PowerShell: .\venv\Scripts\Activate
    # For macOS/Linux/Git Bash: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Execution

1.  **Train and Save the Model:** (Must be done first to create the `.pkl` file)
    ```bash
    python model_trainer.py
    ```
2.  **Launch the Live WebApp:**
    ```bash
    streamlit run app.py
    ```

***

## 🔗 Project Access

| Resource | Link |
| :--- | :--- |
| **Live Web Application** | **https://automated-loan-approval-predictor.streamlit.app/** |
| **Kaggle Dataset** | [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) |

***
