import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("tfidf_vectorizer.pkl", "rb"))  # assuming you used this for TF-IDF
le = pickle.load(open("label_encoder.pkl", "rb"))  # optional if using label encoding


st.title("Automated Essay Scoring")

input_text = st.text_area("Enter your essay here")

if st.button("Predict Score"):
    # Example: dummy features for demonstration
    features = np.zeros((1, 5002))  # 5000 TF-IDF + 2 extra features
    features = scaler.transform(features)
    prediction = model.predict(features)
    st.write("Predicted Score:", prediction)
