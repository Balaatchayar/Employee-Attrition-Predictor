import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/employee_data.csv")
    return df

df = load_data()

# 2. Preprocess
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

X = df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'PerformanceRating']]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. Streamlit UI
st.title("💼 Employee Attrition Predictor")

st.sidebar.header("Enter Employee Details:")
age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
tenure = st.sidebar.slider("Years at Company", 0, 40, 5)
performance = st.sidebar.slider("Performance Rating", 1, 5, 3)

if st.sidebar.button("Predict"):
    input_data = np.array([[age, income, tenure, performance]])
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ This employee is likely to resign. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Probability: {prob:.2f})")

# 4. Visualization
st.subheader("Attrition Insights")
st.bar_chart(df['Attrition'].value_counts())
