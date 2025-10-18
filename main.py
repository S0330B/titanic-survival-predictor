import pandas as pd
import joblib
import streamlit as st 

pipe = joblib.load('decision_tree_classifier.joblib')

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Select your ticket class", ["1st", "2nd", "3rd"])
sex = st.selectbox("Select your gender", ["Male", "Female"])
age = st.number_input("Enter your age", min_value=1, step=1)
sibsp = st.number_input("Enter the number of siblings or spouse", min_value=0, step=1)
parch = st.number_input("Enter the number of parents and children", min_value=0, step=1)
fare = st.number_input("Please enter the fare amount", min_value=1.0, step=0.1)
embarked = st.selectbox("Select where you boarded the ship", ["Cherbourg", "Queenstown", "Southampton"])

if st.button('Predict Survival'):
    try:
        

        pclass_num = int(pclass[0]) 
        sex_num = 1 if sex == "Male" else 2

        query_df = pd.DataFrame([{
            "Pclass": pclass_num,
            "Sex": sex_num,
            "Age": age,
            "Fare": fare
        }])

        predicted_survival = pipe.predict(query_df)[0]



        if predicted_survival == 1:
            st.success(f"🎉 Congrats! You would likely survive.")
        else:
            st.error(f"💀 Sorry, you probably would not survive.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
