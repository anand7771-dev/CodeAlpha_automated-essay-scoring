import streamlit as st
import pickle
import numpy as np

<<<<<<< HEAD
# Load your saved model and vectorizer
model = pickle.load(open("best_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
=======
# Load saved model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("tfidf_vectorizer.pkl", "rb"))  # assuming you used this for TF-IDF
le = pickle.load(open("label_encoder.pkl", "rb"))  # optional if using label encoding

>>>>>>> 7c2d118c35c522e7b69ad1a29c00a9c481ebee1e

st.title("Automated Essay Scoring")

input_text = st.text_area("Enter your essay here")

if st.button("Predict Score"):
    # Transform the input text using TF-IDF
    features = tfidf.transform([input_text])
    
    # Predict using your model
    prediction = model.predict(features)
    
    # Decode label if using label encoder
    if le:
        prediction = le.inverse_transform(prediction)
    
    st.write("Predicted Score:", prediction[0])
