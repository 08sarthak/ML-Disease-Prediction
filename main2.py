# main.py

import streamlit as st
import joblib
import numpy as np

# Import the preprocess_text function from utils.py
from utils import preprocess_text

# Load the trained model and preprocessing objects
clf = joblib.load('disease_prediction_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
le = joblib.load('label_encoder.joblib')

def main():
    st.title("Disease Prediction Based on Symptoms")
    st.write("Enter your symptoms below, and the model will predict the most likely disease.")

    # Text input from user
    user_input = st.text_area("Symptoms Description", "")

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            processed_input = preprocess_text(user_input)
            # Vectorize input text
            input_vector = vectorizer.transform([processed_input])
            # Predict probabilities (if the classifier supports it)
            try:
                probabilities = clf.predict_proba(input_vector)
                # Get top 3 predictions
                top_n = 3
                top_indices = np.argsort(probabilities[0])[-top_n:][::-1]
                diseases = le.inverse_transform(top_indices)
                scores = probabilities[0][top_indices]
                # Display results
                st.success("Top predictions:")
                for i in range(len(diseases)):
                    st.write(f"**{diseases[i]}**: {scores[i]*100:.2f}%")
            except AttributeError:
                # If the classifier does not support predict_proba
                prediction = clf.predict(input_vector)
                predicted_disease = le.inverse_transform(prediction)[0]
                st.success(f"The model predicts: **{predicted_disease}**")
        else:
            st.warning("Please enter your symptoms.")

if __name__ == '__main__':
    main()
