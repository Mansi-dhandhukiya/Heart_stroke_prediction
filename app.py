import streamlit as st
import pandas as pd
import joblib

model = joblib.load("SVM_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart stroke prediction by Mansi")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ["M", "F"])
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure(mm HG)",
    80, 200
)

cholesterol = st.number_input(
    "Fasting Cholesterol(mg/dl)",
    100, 600, 200
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [1, 0]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVM"]
)

max_heart_rate = st.number_input(
    "Max Heart Rate",
    60, 220, 150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


if st.button("Predict"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_blood_sugar,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep exactly the same column order
    input_df = input_df[expected_columns]

    # Make sure everything is numeric
    input_df = input_df.astype(float)

    input_df_scaled = scaler.transform(input_df)

    prediction = model.predict(input_df_scaled)[0]

    if prediction == 1:
        st.error("The patient is likely to have a heart stroke.")
    else:
        st.success("The patient is unlikely to have a heart stroke.")








# import streamlit as st 
# import pandas as pd
# import joblib

# model = joblib.load("SVM_heart.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_columns = joblib.load("columns.pkl")


# st.title("Heart stroke prediction by Mansi")
# st.markdown("Provide the following details")

# age = st.slider("Age",18,100,40)
# sex = st.selectbox("SEX", ["M","F"])
# chest_pain  = st.selectbox("Chest Pain Type", ["ATA","NAP","TA","ASY"])
# resting_bp = st.number_input("Resting Blood Pressure(mm HG)",80,200)
# cholesterol = st.number_input("Fasting Cholesterol(mg/dl)",100,600,200)
# fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1,0])
# resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVM"])
# max_heart_rate = st.number_input("Max Heart Rate",60,220,150)
# exercise_angina = st.selectbox("Exercise Induced Angina", [1,0])
# oldpeak = st.number_input("Old Peak",0.0,6.0,1.0)
# st_slope = st.selectbox("ST Slope", ["Up","Flat","Down"])


# if st.button("Predict"):
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_blood_sugar,
#         'MaxHR': max_heart_rate,
#         'Oldpeak': oldpeak,
#         'Sex'+ sex:1,
#         'ChestPainType'+ chest_pain:1,
#         'RestingECG'+resting_ecg:1,
#         'ExerciseAngina'+exercise_angina :1,
#         'ST_Slope' + st_slope:1
#     }
#     input_df = pd.DataFrame([raw_input])
    
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[expected_columns]
#     input_df_scaled = scaler.transform(input_df)
    
#     prediction = model.predict(input_df_scaled)[0]
    
#     if prediction == 1:
#         st.error("The patient is likely to have a heart stroke.")
#     else:
#         st.success("The patient is unlikely to have a heart stroke.")