import pandas as pd
import random
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import streamlit as st

# -----------------------------
# STEP 1: Generate dataset in memory (1000 employees)
# -----------------------------
education_levels = ['Bachelor', 'Master', 'PhD']
job_roles = ['Software Engineer', 'Data Scientist', 'HR', 'Manager']

rows = []
for _ in range(1000):
    exp = random.randint(0, 20)
    edu = random.choice(education_levels)
    role = random.choice(job_roles)

    # base salary by role
    if role == 'Software Engineer':
        base = 40000
    elif role == 'Data Scientist':
        base = 50000
    elif role == 'HR':
        base = 35000
    else:
        base = 60000

    # education bonus
    if edu == 'Master':
        base += 8000
    elif edu == 'PhD':
        base += 15000

    # experience effect
    salary = base + (exp * random.randint(3000, 6000))
    rows.append([exp, edu, role, salary])

df = pd.DataFrame(rows, columns=['experience', 'education_level', 'job_role', 'salary'])

# -----------------------------
# STEP 2: Train model (no saving)
# -----------------------------
X = df[['experience', 'education_level', 'job_role']]
y = df['salary']

categorical_features = ['education_level', 'job_role']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# -----------------------------
# STEP 3: Streamlit UI
# -----------------------------
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")
st.write("Enter employee details below to predict a fair salary.")

experience = st.slider('Years of Experience', 0, 20, 3)
education_level = st.selectbox('Education Level', ['Bachelor', 'Master', 'PhD'])
job_role = st.selectbox('Job Role', ['Software Engineer', 'Data Scientist', 'HR', 'Manager'])

if st.button('Predict Salary'):
    input_df = pd.DataFrame([[experience, education_level, job_role]],
                            columns=['experience', 'education_level', 'job_role'])
    salary_pred = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: **${salary_pred:,.2f}**")
