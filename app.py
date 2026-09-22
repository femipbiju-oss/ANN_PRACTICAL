import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("employee_performance_ann.keras")

st.title("Employee Performance Prediction")

st.write(
    "Enter the employee's training hours and attendance "
    "to predict performance."
)

# Input fields
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

# Prediction
if st.button("Predict"):

    input_data = np.array([[study_hours, attendance]])

   probability_needs_improvement = model.predict(
    input_data,
    verbose=0
)[0][0]

probability_good = 1 - probability_needs_improvement

if probability_good >= 0.5:
    st.success("Good Performance")
else:
    st.error("Needs Improvement")

st.write(
    f"Probability of Good Performance: {probability_good:.2%}"
)
