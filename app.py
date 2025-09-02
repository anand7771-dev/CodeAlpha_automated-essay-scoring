import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("models/best_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

st.title("Automated Essay Scoring")

input_text = st.text_area("Enter your essay here")

if st.button("Predict Score"):
    features = tfidf.transform([input_text])
    prediction = model.predict(features)
    prediction = le.inverse_transform(prediction)
    st.write("Predicted Score:", prediction[0])
