import streamlit as st
import pickle

# Load your saved model and vectorizer
model = pickle.load(open("models/best_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

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

