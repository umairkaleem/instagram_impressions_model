import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Set page config
st.set_page_config(page_title="Instagram Reach Predictor", layout="wide")

# Load the saved model and vectorizer
@st.cache_resource
def load_assets():
    # Loading the PassiveAggressiveRegressor model
    with open('instagram_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    # Loading the TF-IDF vectorizer for hashtags
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predictor", "About the Model"])

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error("Model files not found! Please ensure 'instagram_model_final.pkl' and 'tfidf_vectorizer.pkl' are in the same folder.")

if page == "Predictor":
    st.title("📊 Instagram Reach Predictor")
    st.write("Enter your post insights to predict total impressions.")

    # Create Input UI
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engagement Metrics")
        likes = st.number_input("Likes", min_value=0, value=100)
        saves = st.number_input("Saves", min_value=0, value=20)
        comments = st.number_input("Comments", min_value=0, value=5)
        shares = st.number_input("Shares", min_value=0, value=2)

    with col2:
        st.subheader("Profile Interactions")
        profile_visits = st.number_input("Profile Visits", min_value=0, value=10)
        follows = st.number_input("Follows Gained", min_value=0, value=1)
        caption = st.text_area("Post Caption", "Enter your post description here...")
        hashtags = st.text_area("Hashtags", "#data #python #machinelearning")

    if st.button("Predict Impressions"):
        # Feature Engineering
        cap_len = len(caption)
        hash_count = len(hashtags.split('#')) - 1

        # Create the input array matching your model's training order:
        # [Likes, Saves, Comments, Shares, Profile Visits, Follows, Caption_Length, Hashtag_Count]
        input_data = np.array([[likes, saves, comments, shares, profile_visits, follows, cap_len, hash_count]])

        prediction = model.predict(input_data)

        st.markdown(f"### 📈 Predicted Total Impressions: **{int(prediction[0]):,}**")
        st.balloons()

else:
    st.title("About the Model")
    st.write("This model uses a **Passive Aggressive Regressor** to predict reach based on early engagement signals.")
    st.info("The prediction is an estimate based on historical data patterns.")