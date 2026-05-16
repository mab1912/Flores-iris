import streamlit as s
import joblib
import numpy as np

# --- Streamlit Page Configuration with Pastel Floral Theme ---
st.set_page_config(
    page_title="Iris Species Predictor",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "This app predicts Iris flower species using a KNN model."
    }
)

st.markdown("""
    <style>
    /* Main background with a subtle floral touch */
    .stApp {
        background-color: #F8F4F9; /* Very light lavender/pink */
        background-image: url('https://www.transparenttextures.com/patterns/white-diamond.png'); /* Subtle texture */
        background-size: 100px;
    }
    /* Sidebar background */
    .st-emotion-cache-1ldf0gi.eczjsme4 {
        background-color: #E6E0EC; /* Slightly darker pastel lavender */
        border-right: 1px solid #D5C2E0;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #6C4A7C; /* Deep lavender */
        font-family: 'Georgia', serif;
    }
    /* Text color */
    .st-emotion-cache-zt5ig8.e1nzilvr3 {
        color: #4A4A4A;
    }
    /* Buttons */
    .stButton>button {
        background-color: #9370DB; /* Medium Purple */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #8A2BE2; /* Blue Violet */
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #B0C4DE; /* Light steel blue */
    }
    .stSlider [data-baseweb="slider"] {
        background-color: #DDA0DD; /* Plum */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the models (assuming they are in the current working directory)
try:
    best_knn_model = joblib.load('best_knn_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Error: Model files (best_knn_model.joblib or label_encoder.joblib) not found. Make sure they are in the expected directory.")
    st.stop()

st.title("🌸 Iris Species Predictor 🌸")
st.markdown("### Discover the species of an Iris flower based on its physical characteristics.")

st.write("
") # Add some space

with st.sidebar:
    st.header("Flower Measurements")
    st.markdown("Adjust the sliders below to describe your Iris flower.")

    # Input features in sidebar for better UX
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, key='sl')
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, key='sw')
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.3, key='pl')
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2, key='pw')


st.subheader("Prediction Results")

# Prediction button in the main area
if st.button('Predict Species', help='Click to get the prediction!'):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction_encoded = best_knn_model.predict(input_features)
    predicted_species = label_encoder.inverse_transform(prediction_encoded)

    st.success(f"The predicted Iris species is: ✨ **{predicted_species[0].capitalize()}** ✨")
    st.write("Thank you for using the Iris Species Predictor!")

st.markdown("""
---
<p style='text-align: center; font-size: 14px; color: #888;'>
    Model: K-Nearest Neighbors | Data: Iris Dataset <br>
    Created with Streamlit and powered by Colab.
</p>
