import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="InstaReach AI", 
    page_icon="📸",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        font-size: 30px;
        font-weight: bold;
        color: #00FFCC;
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #00FFCC;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    # 1. Load the Brain (Model) and the Translator (Scaler)
    model = pickle.load(open('instagram_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    
    # 2. Load the cleaned data for account stats
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

# --- Sidebar ---
st.sidebar.title("InstaReach AI")
st.sidebar.markdown(f"**Total Posts Analyzed:** {avg_stats['total_posts']}")
st.sidebar.info("Model Accuracy: 78.24%")

# --- Main UI ---
st.title("📸 Instagram Reach Predictor")
st.write("Simulate post performance using our high-accuracy Random Forest model.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Content Strategy")
    caption = st.text_area("Post Caption", placeholder="Write your caption here...", height=150)
    hashtags = st.text_area("Hashtags", placeholder="#marketing #analytics", height=100)
    
with col2:
    st.subheader("⚙️ Engagement Settings")
    likes = st.slider("Likes", 0, 5000, int(avg_stats['likes']))
    saves = st.slider("Saves", 0, 1000, int(avg_stats['saves']))
    shares = st.slider("Shares", 0, 500, int(avg_stats['shares']))
    
    with st.expander("Advanced Metrics"):
        comments = st.number_input("Comments", value=int(avg_stats['comments']))
        visits = st.number_input("Profile Visits", value=int(avg_stats['visits']))
        follows = st.number_input("Follows", value=int(avg_stats['follows']))

st.markdown("---")

if st.button("Calculate Predicted Reach"):
    with st.spinner('AI analyzing signals...'):
        # 1. Engineering features
        cap_len = len(caption)
        hash_count = hashtags.count('#')
        
        # 2. Create input array (Matches training order)
        # Order: Likes, Saves, Comments, Shares, Profile Visits, Follows, Cap_Len, Hash_Count
        input_raw = np.array([[likes, saves, comments, shares, visits, follows, cap_len, hash_count]])
        
        # 3. Apply the Scaler (Fixes the math)
        input_scaled = scaler.transform(input_raw)
        
        # 4. Predict (In Log format)
        pred_log = model.predict(input_scaled)
        
        # 5. Reverse Log (Back to real impressions)
        final_reach = np.expm1(pred_log)[0]
        
        # 6. Display Results
        st.balloons()
        st.markdown(f"<div class='prediction-box'>Predicted Reach: {int(final_reach):,} Impressions</div>", unsafe_allow_html=True)

# --- Analytics Section ---
st.markdown("---")
st.header("📊 Account Context")
st.write("How your current inputs compare to your account history:")
c1, c2, c3 = st.columns(3)
c1.metric("Avg Likes", f"{int(avg_stats['likes'])}")
c2.metric("Avg Saves", f"{int(avg_stats['saves'])}")
c3.metric("Avg Shares", f"{int(avg_stats['shares'])}")