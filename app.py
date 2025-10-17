import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/employee_data.csv")
    return df

df = load_data()

# -------------------------------
# 2. Preprocess
# -------------------------------
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

X = df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'PerformanceRating']]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("💼 Employee Attrition Predictor")

st.sidebar.header("Enter Employee Details:")
age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
tenure = st.sidebar.slider("Years at Company", 0, 40, 5)
performance = st.sidebar.slider("Performance Rating", 1, 5, 3)

# Display Model Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.markdown(f"**Model Accuracy:** `{accuracy:.2f}`")

# -------------------------------
# 4. Prediction Section
# -------------------------------
if st.sidebar.button("Predict"):
    input_data = np.array([[age, income, tenure, performance]])
    prediction = model.predict(input_data)
    prob_attrition = model.predict_proba(input_data)[0][1]
    prob_stay = 1 - prob_attrition

    if prediction[0] == 1:
        st.error(f"⚠️ This employee is likely to **resign**. (Probability: {prob_attrition:.2f})")
    else:
        st.success(f"✅ This employee is likely to **stay**. (Probability: {prob_stay:.2f})")

# -------------------------------
# 5. Attrition Insights
# -------------------------------
st.subheader("📊 Attrition Insights")
st.bar_chart(df['Attrition'].value_counts())

with st.expander("📘 View Sample Dataset"):
    st.dataframe(df.head())

# -------------------------------
# 6. Feature Importance Visualization
# -------------------------------
st.subheader("🔍 Feature Importance Analysis")

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

st.caption("This chart shows which factors most influence employee attrition predictions.")
