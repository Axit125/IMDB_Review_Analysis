import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved models
logreg_model = joblib.load('models/logreg_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Initialize the vectorizer (ensure it's the same one used in training)
tfidf_vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')

# Define a helper function for making predictions
def predict_sentiment(text, model):
    text_tfidf = tfidf_vectorizer.transform([text])
    return model.predict(text_tfidf)[0]

# Streamlit app setup
st.title('Sentiment Analysis with Multiple Models')
st.markdown('This app allows you to predict sentiment using different models (Logistic Regression, Random Forest, XGBoost).')

# Text input for prediction
input_text = st.text_area("Enter text for sentiment analysis:", "I love this product!")

# Model selection dropdown
model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest", "XGBoost"])

# Prediction logic
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = predict_sentiment(input_text, logreg_model)
    elif model_choice == "Random Forest":
        prediction = predict_sentiment(input_text, rf_model)
    else:
        prediction = predict_sentiment(input_text, xgb_model)
    
    result = 'Positive' if prediction == 1 else 'Negative'
    st.write(f"Prediction: {result}")
