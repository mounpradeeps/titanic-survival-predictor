import streamlit as st
import pandas as pd
import joblib
model = joblib.load("titanic_final_pipeline.pkl")
st.title("🚢 Titanic Survival Predictor — CI/CD Demo")
st.write("Enter passenger details to predict survival.")
st.caption("CD test: version 2")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex_choice = st.selectbox("Sex", ["Male", "Female"])
sex = 0 if sex_choice == "Male" else 1
title = st.selectbox(
    "Title",
    ["Mr", "Miss", "Mrs", "Master", "Rare"]
)
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=7.25, step=1.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
if st.button("Predict"):
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Title": title    
    }])
    prediction = model.predict(passenger)[0]
    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")
