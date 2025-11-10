import streamlit as st
import pandas as pd
import joblib

st.title('🎵 Voice Command Classification App')
st.write('This app uses a Logistic Regression model to classify music genres (Rock, Pop, Jazz, Classical).')
st.write('Use the sliders below to simulate the song features.')

# Load the trained model
try:
    model = joblib.load('genre_model.pkl')
except FileNotFoundError:
    st.error("Model file (genre_model.pkl) not found. Please run the 'train_model.py' script first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Sidebar for feature inputs
st.sidebar.header('Input Features')
st.sidebar.write('Simulate the song features here:')

tempo = st.sidebar.slider('Tempo (BPM)', 60, 200, 120)
loudness = st.sidebar.slider('Instrument Loudness (dB)', 50, 120, 90)
pitch = st.sidebar.slider('Vocal Pitch (Hz)', 100, 600, 300)

# Convert user inputs to DataFrame
input_data = pd.DataFrame({
    'tempo': [tempo],
    'instrument_loudness_db': [loudness],
    'vocal_pitch_hz': [pitch]
})

# Display input data on main page
st.subheader('Selected Features:')
st.write(input_data)

# Prediction button
if st.button('Classify Genre'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader('Prediction Result:')
    st.success(f'Predicted Genre: **{prediction[0].capitalize()}**')
    
    st.write('Prediction Probabilities (Model confidence):')
    probs_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.dataframe(probs_df.style.highlight_max(axis=1))

st.info("Note: This model is currently trained on 1000samples.")
