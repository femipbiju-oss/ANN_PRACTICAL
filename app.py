import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd


# --------------------------------------------------
# Load trained ANN model
# --------------------------------------------------

model = tf.keras.models.load_model(
    "employee_performance_ann.keras"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Employee Performance Prediction")

st.write(
    "Enter the employee's training hours and attendance "
    "to predict performance."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

study_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    # Create input data
    input_data = np.array([
        [study_hours, attendance]
    ])

    # Get model output
    probability_needs_improvement = model.predict(
        input_data,
        verbose=0
    )[0][0]

    # Since the model output represents
    # Needs Improvement probability
    probability_good = 1 - probability_needs_improvement

    # Classification
    if probability_good >= 0.5:

        st.success("Good Performance")

    else:

        st.error("Needs Improvement")

    # Display probability
    st.write(
        f"Probability of Good Performance: "
        f"{probability_good:.2%}"
    )
