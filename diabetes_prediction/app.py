
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Type 2 Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F8F5;
}

h1 {
    color: #176B55;
    text-align: center;
}

h2, h3 {
    color: #176B55;
}

.stButton > button {
    background-color: #176B55;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 17px;
}

.stButton > button:hover {
    background-color: #0E5140;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED FILES ----------------

@st.cache_resource
def load_files():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder


st.title("🩺 Type 2 Diabetes Risk Prediction")
st.write(
    "Enter the person's health details to predict "
    "the diabetes classification."
)

try:
    model, scaler, encoder = load_files()
except Exception as e:
    st.error(
        "Unable to load the model files. Make sure "
        "best_model.pkl, scaler.pkl and encoder.pkl "
        "are in the same folder as app.py."
    )
    st.code(str(e))
    st.stop()


# ---------------- USER INPUTS ----------------

st.subheader("Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

with col2:
    age = st.number_input(
        "Age (years)",
        min_value=0.0,
        max_value=120.0,
        value=30.0,
        step=1.0
    )

st.subheader("Medical Information")

col1, col2 = st.columns(2)

with col1:
    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        ["No", "Yes"]
    )

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "No Info",
            "current",
            "former",
            "ever",
            "not current"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    hba1c_level = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=20.0,
        value=5.5,
        step=0.1
    )

with col2:
    blood_glucose_level = st.number_input(
        "Blood Glucose Level",
        min_value=50.0,
        max_value=500.0,
        value=100.0,
        step=1.0
    )


# ---------------- PREPROCESS INPUT ----------------

def prepare_input(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c_level,
    blood_glucose_level,
    encoder,
    scaler
):
    data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    # Handle encoders saved as a dictionary
    if isinstance(encoder, dict):
        for column in ["gender", "smoking_history"]:
            if column in encoder:
                enc = encoder[column]
                value = data[column].astype(str)

                if hasattr(enc, "classes_"):
                    known = list(enc.classes_)

                    # Map unseen categories to a known category
                    if value.iloc[0] not in known:
                        value.iloc[0] = (
                            "No Info"
                            if "No Info" in known
                            else known[0]
                        )

                data[column] = enc.transform(value)

    # Handle an encoder trained on both categorical columns
    elif hasattr(encoder, "categories_"):
        categorical = data[
            ["gender", "smoking_history"]
        ].copy()

        encoded = encoder.transform(categorical)

        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()

        numeric = data[
            [
                "age",
                "hypertension",
                "heart_disease",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level"
            ]
        ].to_numpy()

        data = np.concatenate(
            [numeric, encoded],
            axis=1
        )

        return scaler.transform(data)

    # Handle a single LabelEncoder
    elif hasattr(encoder, "classes_"):
        for column in ["gender", "smoking_history"]:
            value = str(data[column].iloc[0])

            if value not in encoder.classes_:
                raise ValueError(
                    "The saved LabelEncoder does not contain "
                    f"the category {value!r} for {column}. "
                    "Use the encoders created during training."
                )

            data[column] = encoder.transform([value])[0]

    # Ensure the model receives features in training order
    if hasattr(scaler, "feature_names_in_"):
        data = data[list(scaler.feature_names_in_)]

    return scaler.transform(data)


# ---------------- PREDICTION ----------------

if st.button("Predict Diabetes Risk"):

    try:
        input_data = prepare_input(
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c_level,
            blood_glucose_level,
            encoder,
            scaler
        )

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.error(
                "⚠️ The model classified this input "
                "as Diabetes Positive."
            )
            st.write(
                "Please consult a qualified healthcare "
                "professional for proper assessment."
            )

        else:
            st.success(
                "✅ The model classified this input "
                "as Diabetes Negative."
            )

        # Display probability if supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                diabetes_probability = probabilities[
                    classes.index(1)
                ]

                st.metric(
                    "Model-estimated diabetes probability",
                    f"{diabetes_probability * 100:.2f}%"
                )

                st.caption(
                    "This is a model estimate, not a clinical "
                    "diagnosis or a validated individual risk score."
                )

    except Exception as e:
        st.error(
            "Prediction failed. The preprocessing used here "
            "must match the preprocessing used to train "
            "your saved model."
        )
        st.code(str(e))

st.divider()

st.caption(
    "Educational project only. This tool does not diagnose "
    "diabetes or replace medical advice."
)
