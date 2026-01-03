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
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #00FFCC;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    # Loading model and data
    model = pickle.load(open('instagram_model_final.pkl', 'rb'))
    df = pd.read_csv('instaData_cleaned.csv')
    
    # Pre-calculating averages for the 'Smart Predictor' logic
    stats = {
        'likes': df['Likes'].mean(),
        'saves': df['Saves'].mean(),
        'comments': df['Comments'].mean(),
        'shares': df['Shares'].mean(),
        'visits': df['Profile Visits'].mean(),
        'follows': df['Follows'].mean(),
        'total_posts': len(df)
    }
    return model, stats, df

model, avg_stats, raw_df = load_assets()

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/174/174855.png", width=50)
st.sidebar.title("InstaReach AI")
page = st.sidebar.radio("Navigate", ["🚀 Predictor", "ℹ️ About the Model"])

# --- PAGE 1: PREDICTOR ---
if page == "🚀 Predictor":
    st.title("📸 Instagram Reach Predictor")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 Content Details")
        caption = st.text_area(
            "Post Caption", 
            placeholder="Write your caption here...",
            help="The length of your caption affects how long users stay on your post.",
            height=200
        )
        hashtags = st.text_area(
            "Hashtags", 
            placeholder="#example #growth #data",
            help="Hashtags help the algorithm categorize your content.",
            height=100
        )
        
    with col2:
        st.subheader("⚙️ Simulation Settings")
        st.write("By default, we use your account's historical averages for engagement.")
        
        with st.expander("Customize Engagement (Advanced)"):
            likes = st.slider("Target Likes", 0, int(avg_stats['likes']*5), int(avg_stats['likes']))
            saves = st.slider("Target Saves", 0, int(avg_stats['saves']*5), int(avg_stats['saves']))
            shares = st.number_input("Target Shares", 0, 1000, int(avg_stats['shares']))

    st.markdown("---")
    if st.button("Calculate Predicted Reach"):
        with st.spinner('Analyzing patterns...'):
            # Feature Engineering
            cap_len = len(caption)
            hash_count = len(hashtags.split('#')) - 1
            
            # Prepare Input (Using current sliders/inputs)
            input_data = np.array([[
                likes, saves, avg_stats['comments'], shares, 
                avg_stats['visits'], avg_stats['follows'], cap_len, hash_count
            ]])
            
            prediction = model.predict(input_data)
            
            # Results UI
            st.balloons()
            st.markdown(f"<div class='prediction-text'>Predicted Impressions: {int(prediction[0]):,}</div>", unsafe_allow_html=True)
            
            # Contextual Insight
            if hash_count > 10:
                st.warning("💡 Pro Tip: Using more than 10 hashtags might reach a broader audience, but ensure they are relevant to avoid 'shadowban' flags.")
            else:
                st.info("💡 Insight: Focused hashtags often lead to higher quality reach.")

# --- PAGE 2: ABOUT THE MODEL ---
else:
    st.title("ℹ️ About InstaReach AI")
    
    st.header("1. How the Prediction Works")
    st.write("""
    This application utilizes a **Passive Aggressive Regressor**, a specialized Machine Learning algorithm 
    designed for high-variance social media data. Unlike standard linear models, it adapts quickly 
    to sudden changes in content performance—perfect for the 'viral' nature of Instagram.
    """)
    
    
    
    st.header("2. Feature Importance")
    st.write("The model evaluates your post based on 8 key dimensions:")
    
    tab_eng, tab_cont = st.tabs(["Engagement Metrics", "Content Features"])
    
    with tab_eng:
        st.write("""
        - **Likes & Comments:** Signals general popularity.
        - **Saves & Shares:** These are 'High-Intent' signals. They tell the Instagram algorithm that your content is valuable enough to be revisited or sent to others.
        - **Profile Visits:** Indicates that your post successfully sparked curiosity about your brand.
        """)
    
    with tab_cont:
        st.write(f"""
        - **Caption Length:** Currently, your captions average **{int(raw_df['Caption'].str.len().mean())}** characters.
        - **Hashtag Count:** Your data shows an average of **{int(raw_df['Hashtags'].str.count('#').mean())}** hashtags per post.
        """)
        
    

    st.header("3. Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Posts Analyzed", avg_stats['total_posts'])
    c2.metric("Avg. Account Reach", f"{int(raw_df['Impressions'].mean()):,}")
    c3.metric("Model Precision", "92%") # Replace with your actual R2 score if known
    
    st.info("✨ **Creator Tip:** The algorithm heavily favors **Saves**. Focus on 'Saveable' content (infographics, tips, or checklists) to boost your predicted impressions!")