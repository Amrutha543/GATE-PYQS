import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("GATE Question Topic Classifier")

question = st.text_area("Enter your GATE question:")

if st.button("Predict Topic"):
    if question:
        q_vec = vectorizer.transform([question])
        prediction = model.predict(q_vec)
        st.success(f"Predicted Topic: {prediction[0]}")
    else:
        st.warning("Please enter a question.")