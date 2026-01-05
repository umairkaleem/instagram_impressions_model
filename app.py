import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="InstaReach AI", 
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-text {
        font-size: 32px;
        font-weight: bold;
        color: #00FFCC;
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #00FFCC;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    # Load the trained Random Forest model and the Scaler
    model = pickle.load(open('instagram_model_final.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    
    # Load cleaned data for stats
    df = pd.read_csv('instaData_cleaned.csv')
    
    stats = {
        'likes': df['Likes'].mean(),
        'saves': df['Saves'].mean(),
        'comments': df['Comments'].mean(),
        'shares': df['Shares'].mean(),
        'visits': df['Profile Visits'].mean(),
        'follows': df['Follows'].mean(),
        'total_posts': len(df)
    }
    return model, scaler, stats, df

model, scaler, avg_stats, raw_df = load_assets()

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/174/174855.png", width=50)
st.sidebar.title("InstaReach AI")
page = st.sidebar.radio("Navigate", ["🚀 Predictor", "ℹ️ About the Model"])

# --- PAGE 1: PREDICTOR ---
if page == "🚀 Predictor":
    st.title("📸 Instagram Reach Predictor")
    st.markdown("Predict your post's total impressions using AI trained on engagement patterns.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 Content Strategy")
        caption = st.text_area(
            "Post Caption", 
            placeholder="What will your post say?",
            help="Caption length influences user dwell time.",
            height=150
        )
        hashtags = st.text_area(
            "Hashtags", 
            placeholder="#data #ai #growth",
            help="Include the '#' symbol for each hashtag.",
            height=100
        )
        
    with col2:
        st.subheader("⚙️ Engagement Simulation")
        st.write("Adjust the sliders to see how engagement affects reach.")
        
        likes = st.slider("Predicted Likes", 0, 5000, int(avg_stats['likes']))
        saves = st.slider("Predicted Saves", 0, 2000, int(avg_stats['saves']))
        shares = st.number_input("Predicted Shares", 0, 1000, int(avg_stats['shares']))
        
        with st.expander("Advanced Metrics"):
            comments = st.number_input("Comments", 0, 500, int(avg_stats['comments']))
            profile_visits = st.number_input("Profile Visits", 0, 1000, int(avg_stats['visits']))
            follows = st.number_input("New Follows", 0, 100, int(avg_stats['follows']))

    st.markdown("---")
    
    if st.button("Calculate Predicted Reach"):
        with st.spinner('AI is analyzing content signals...'):
            # 1. Feature Engineering from user input
            cap_len = len(caption)
            hash_count = hashtags.count('#')
            
            # 2. Prepare Input Data (Must match the 8 features used in training)
            # Order: Likes, Saves, Comments, Shares, Profile Visits, Follows, Caption_Len, Hashtag_Count
            input_features = np.array([[
                likes, saves, comments, shares, 
                profile_visits, follows, cap_len, hash_count
            ]])
            
            # 3. SCALE THE DATA (Crucial: Uses the saved scaler.pkl)
            input_scaled = scaler.transform(input_features)
            
            # 4. PREDICT
            prediction_log = model.predict(input_scaled)
            
            # 5. REVERSE LOG TRANSFORMATION (np.expm1)
            final_reach = np.expm1(prediction_log)[0]
            
            # Results UI
            st.balloons()
            st.markdown(f"<div class='prediction-text'>Predicted Impressions: {int(final_reach):,}</div>", unsafe_allow_html=True)
            
            # Insights
            st.info(f"💡 **Analysis:** Your caption length ({cap_len} chars) and hashtag usage ({hash_count} tags) were factored into this prediction.")

# --- PAGE 2: ABOUT THE MODEL ---
else:
    st.title("ℹ️ About the Technology")
    
    st.header("1. The Algorithm")
    st.write("""
    This app uses a **Random Forest Regressor**. This model is an 'Ensemble' method, 
    meaning it combines the predictions of 100 different decision trees to provide 
    a stable and accurate estimate of reach, even with high-variance social media data.
    """)
    
    st.header("2. Data Preprocessing")
    st.write("""
    - **Outlier Stabilization:** Viral 'noise' was replaced with median values to ensure predictions are realistic for everyday content.
    - **Log Transformation:** We use Logarithmic scaling to handle the exponential growth nature of social media.
    - **Feature Scaling:** All inputs are standardized using a `StandardScaler` to ensure metrics like 'Likes' and 'Caption Length' are weighted correctly.
    """)

    st.header("3. Dataset Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Posts Analyzed", avg_stats['total_posts'])
    c2.metric("Median Reach", f"{int(raw_df['Impressions'].median()):,}")
    c3.metric("Model Stability", "High (Random Forest)")
    
    st.success("✨ **Pro Tip:** Focus on increasing **Saves**. The model identifies 'Saves' as a primary trigger for the Instagram Explore algorithm!")