import pickle
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Title and Image
st.title("Stock Market Health Analysis and Fraud Detection")
st.image("image.jpg")

# User Input
input_df = st.text_input("Please provide all the required feature details, separated by commas:")

submit = st.button("Submit")

if submit:
    try:
        # Load Model
        model = pickle.load(open('model_rf.pkl', 'rb'))
        
        # Process Input
        input_df_split = input_df.split(',')
        features = np.asarray(input_df_split, dtype=np.float64)
        
        # Check if feature count matches
        expected_feature_count = model.n_features_in_
        if features.shape[0] != expected_feature_count:
            st.error(f"Expected {expected_feature_count} features, but got {features.shape[0]}. Please recheck your input.")
        else:
            # Prediction
            prediction = model.predict(features.reshape(1, -1))

            # Output Result
            if prediction[0] == "Not Fraud":
                st.write("Not a Fraudulent")
            else:
                st.write("Fraudulent")
    except ValueError as ve:
        st.error(f"Input error: {str(ve)}. Please ensure all features are numeric and formatted correctly.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
