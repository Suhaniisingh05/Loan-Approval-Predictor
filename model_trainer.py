import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Configuration ---
FILE_PATH = 'train.csv'
MODEL_FILE_NAME = 'loan_prediction_model.pkl'

# --- 1. Load Data ---
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found. Please download the dataset and place it in the project folder.")
    exit()

# --- 2. Data Cleaning and Imputation ---

# Impute Categorical Columns (using Mode)
# Important: We are using the simplest method (mode) for initial learning.
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Impute Numerical Columns (using Mean)
for col in ['LoanAmount', 'Loan_Amount_Term']:
    df[col].fillna(df[col].mean(), inplace=True)

# --- 3. Feature Engineering and Preprocessing ---

# Feature Engineering: Create Total Income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)

# Define X (Features) and y (Target)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Drop the ID column
X.drop('Loan_ID', axis=1, inplace=True)

# Convert all remaining categorical features to numerical using One-Hot Encoding
# drop_first=True prevents multicollinearity.
X = pd.get_dummies(X, drop_first=True)

# Convert the target variable 'Loan_Status' (Y/N) to numerical (1/0)
# 'Y' (Approved) will be mapped to 1, 'N' (Rejected) to 0
le = LabelEncoder()
y = le.fit_transform(y) 

# --- 4. Model Training ---

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Optional: Print evaluation metrics (for debugging/validation)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f"Model Training Complete. Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- 5. Save Model and Column Names ---
# Saving column names is VITAL for Streamlit to know the input order.
model_data = {
    'model': model,
    'columns': X.columns.tolist()
}

with open(MODEL_FILE_NAME, 'wb') as file:
    pickle.dump(model_data, file)

print(f"\nSuccessfully created and saved the model to {MODEL_FILE_NAME}")