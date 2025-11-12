import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# configure
FILE_PATH = ''
MODEL_FILE_PATH =''

# load data
try:
  df = pd.readcsv(FILE_PATH)
except FileNotFoundError:
  exit()

# data cleaning and imputation
  # Impute Categorical Columns (using Mode)
for col in ['Gender','Married','Dependents','self_Employed','Credit_History']:
    df[col].fillna(df[col].mode()[0] , inplace = true)

for col in ['LoanAmount','Loan_Amount_Term']:
    df[col].fillna(df[col].mean()[0] , inplace = true)

# Feature Engineering and Preprocessing

# Feature engineering - Total income
df['Total_income'] = df['Applicants_income'] + df['CoApplicants_income']
df.drop(['Applicants_income','CoApplicants_income'],axis =1 , inplace = True)

# Define X(Features) and Y(Target Varible)

X = df.drop('Loan_Status',axis = 1)
Y = df['Loan_Status']

X.drop('Loan_id',axis =1 ,inplace = True)

# Convert all remaining categorical features to numerical using One-Hot Encoding
# drop_first=True prevents multicollinearity.

X = pd.get_dummies(X,drop_first=True)

# Convert the target variable 'Loan_Status' (Y/N) to numerical (1/0)
# 'Y' (Approved) will be mapped to 1, 'N' (Rejected) to 0
le = LabelEncoder()
Y = le.fit_transform(Y)

# --- 4. Model Training ---

# split the training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2,random_state=42)

# train the model
model = LogisticRegression(solver = 'liblinear',random_state=42)
model.fit(X_train,Y_train)

