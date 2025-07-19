import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("student_scores.csv")

# Train model
X = data[['Hours']]
y = data['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎓 Student Marks Predictor")
st.write("Enter the number of hours you studied to predict your score:")

# Input slider
hours = st.slider("Hours Studied", 0.0, 10.0, step=0.25)

# Prediction
if st.button("Predict Marks"):
    predicted = model.predict([[hours]])
    st.success(f"📈 Predicted Score: **{predicted[0]:.2f}** out of 100")

# Optional: Show dataset
with st.expander("📊 Show Training Dataset"):
    st.dataframe(data)

# Optional: Add line chart
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(data['Hours'], data['Scores'], color='blue')
ax.plot(X, model.coef_ * X + model.intercept_, color='red')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Marks Scored')
ax.set_title('Study Hours vs Marks')
st.pyplot(fig)
