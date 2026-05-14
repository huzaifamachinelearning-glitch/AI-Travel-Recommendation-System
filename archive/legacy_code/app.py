import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import os

# Page configuration
st.set_page_config(
    page_title="AI Travel Recommendation System",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_resources():
    try:
        df = pd.read_csv('data/processed_dataset.csv')
        
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return df, model, encoders, scaler
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None

# Initialize Groq client
@st.cache_resource
def init_groq():
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        return Groq(api_key=api_key)
    return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌏 AI-Powered Travel Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover Your Perfect Indian Destination with AI</p>', unsafe_allow_html=True)
    
    # Load resources
    df, model, encoders, scaler = load_resources()
    groq_client = init_groq()
    
    if df is None:
        st.error("⚠️ Please run preprocessing and training scripts first!")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Smart Search", "💬 AI Chatbot", "📊 Analytics", "ℹ️ About"])
    
    # Page routing
    if page == "🏠 Home":
        home_page(df)
    elif page == "🔍 Smart Search":
        smart_search_page(df, model, encoders, scaler)
    elif page == "💬 AI Chatbot":
        chatbot_page(df, groq_client)
    elif page == "📊 Analytics":
        analytics_page(df)
    elif page == "ℹ️ About":
        about_page()

def home_page(df):
    st.header("🎯 Welcome to Your Travel Companion!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Total Destinations", df['place_name'].nunique())
    
    with col2:
        st.metric("⭐ Avg Rating", f"{df['rating_given'].mean():.2f}")
    
    with col3:
        st.metric("💰 Avg Budget", f"₹{df['budget_thousand_inr'].mean():.0f}k")
    
    with col4:
        st.metric("📅 Avg Trip Duration", f"{df['trip_duration_days'].mean():.1f} days")
    
    st.markdown("---")
    
    # Featured destinations
    st.subheader("🌟 Top Rated Destinations")
    
    top_places = df.groupby('place_name').agg({
        'rating_given': 'mean',
        'place_type': 'first',
        'state': 'first',
        'budget_thousand_inr': 'mean',
        'safety_index': 'mean'
    }).sort_values('rating_given', ascending=False).head(6)
    
    cols = st.columns(3)
    for idx, (place, data) in enumerate(top_places.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>📍 {place}</h3>
                <p><strong>State:</strong> {data['state']}</p>
                <p><strong>Type:</strong> {data['place_type']}</p>
                <p><strong>Rating:</strong> ⭐ {data['rating_given']:.2f}</p>
                <p><strong>Budget:</strong> ₹{data['budget_thousand_inr']:.0f}k/day</p>
                <p><strong>Safety:</strong> 🛡️ {data['safety_index']:.1f}/5</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Destinations by Type")
        place_type_counts = df['place_type'].value_counts()
        fig = px.pie(values=place_type_counts.values, names=place_type_counts.index, 
                     hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Travel by Season")
        season_counts = df['season'].value_counts()
        fig = px.bar(x=season_counts.index, y=season_counts.values,
                     color=season_counts.values, color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title="Season", yaxis_title="Number of Trips")
        st.plotly_chart(fig, use_container_width=True)

def smart_search_page(df, model, encoders, scaler):
    st.header("🔍 Smart Destination Search")
    st.write("Fill in your preferences and let our AI recommend the perfect destinations!")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.slider("💰 Budget (₹ in thousands per day)", 5, 50, 20)
        duration = st.slider("📅 Trip Duration (days)", 1, 15, 5)
        place_type = st.selectbox("🏞️ Destination Type", df['place_type'].unique())
    
    with col2:
        companions = st.selectbox("👥 Travel Companions", df['companions'].unique())
        interest = st.selectbox("🎯 Interest", df['interest'].unique())
        season = st.selectbox("🌤️ Season", df['season'].unique())
    
    with col3:
        accessibility = st.selectbox("🚶 Accessibility", ['Easy', 'Moderate', 'Hard'])
        climate = st.selectbox("🌡️ Climate Preference", df['climate'].unique())
        min_safety = st.slider("🛡️ Minimum Safety Index", 1.0, 5.0, 3.0, 0.5)
    
    if st.button("🔎 Find Perfect Destinations", type="primary"):
        with st.spinner("🤖 AI is analyzing your preferences..."):
            # Filter destinations
            filtered_df = df[
                (df['budget_thousand_inr'] <= budget) &
                (df['trip_duration_days'] <= duration) &
                (df['place_type'] == place_type) &
                (df['companions'] == companions) &
                (df['interest'] == interest) &
                (df['season'] == season) &
                (df['accessibility'] == accessibility) &
                (df['safety_index'] >= min_safety)
            ]
            
            if len(filtered_df) == 0:
                st.warning("⚠️ No exact matches found. Showing closest alternatives...")
                filtered_df = df[
                    (df['budget_thousand_inr'] <= budget + 10) &
                    (df['place_type'] == place_type)
                ].head(20)
            
            # Get top recommendations
            recommendations = filtered_df.groupby('place_name').agg({
                'rating_given': 'mean',
                'state': 'first',
                'place_type': 'first',
                'budget_thousand_inr': 'mean',
                'safety_index': 'mean',
                'popularity_level': 'mean',
                'latitude': 'mean',
                'longitude': 'mean'
            }).sort_values('rating_given', ascending=False).head(5)
            
            st.success(f"✅ Found {len(recommendations)} perfect destinations for you!")
            
            # Display recommendations
            for idx, (place, data) in enumerate(recommendations.iterrows(), 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h2>#{idx} 📍 {place}</h2>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <p><strong>📍 Location:</strong> {data['state']}</p>
                            <p><strong>🏞️ Type:</strong> {data['place_type']}</p>
                            <p><strong>⭐ Rating:</strong> {data['rating_given']:.2f}/5</p>
                        </div>
                        <div>
                            <p><strong>💰 Budget:</strong> ₹{data['budget_thousand_inr']:.0f}k/day</p>
                            <p><strong>🛡️ Safety:</strong> {data['safety_index']:.1f}/5</p>
                            <p><strong>🔥 Popularity:</strong> {data['popularity_level']:.1f}/5</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Map visualization
            st.subheader("🗺️ Recommended Destinations on Map")
            fig = px.scatter_mapbox(
                recommendations.reset_index(),
                lat='latitude',
                lon='longitude',
                hover_name='place_name',
                hover_data={'rating_given': ':.2f', 'safety_index': ':.1f'},
                color='rating_given',
                size='popularity_level',
                color_continuous_scale='Viridis',
                zoom=4,
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)

def chatbot_page(df, groq_client):
    st.header("💬 AI Travel Assistant")
    st.write("Ask me anything about travel destinations in India!")
    
    if not groq_client:
        st.warning("⚠️ AI Chatbot requires GROQ_API_KEY environment variable.")
        st.info("Get free API key from: https://console.groq.com")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about travel destinations..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get context from data
                    top_places = df.groupby('place_name')['rating_given'].mean().sort_values(ascending=False).head(5)
                    context = f"Top destinations: {', '.join(top_places.index.tolist())}"
                    
                    # Call Groq API
                    messages = [
                        {"role": "system", "content": f"You are a helpful Indian travel expert. {context}"},
                        *st.session_state.messages
                    ]
                    
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024
                    )
                    
                    ai_response = response.choices[0].message.content
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    st.error(f"Error: {e}")

def analytics_page(df):
    st.header("📊 Travel Analytics Dashboard")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trips", len(df))
    with col2:
        st.metric("Unique Destinations", df['place_name'].nunique())
    with col3:
        st.metric("Avg Rating", f"{df['rating_given'].mean():.2f}")
    with col4:
        st.metric("High Ratings %", f"{(df['high_rating'].sum()/len(df)*100):.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎯 Preferences", "💰 Budget Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Seasonal Travel Trends")
            season_data = df.groupby('season').size()
            fig = px.bar(x=season_data.index, y=season_data.values, 
                        color=season_data.values, color_continuous_scale='Sunset')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Monthly Distribution")
            month_data = df['travel_month'].value_counts().sort_index()
            fig = px.line(x=month_data.index, y=month_data.values, markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Travel Companions")
            companion_data = df['companions'].value_counts()
            fig = px.pie(values=companion_data.values, names=companion_data.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Interest Categories")
            interest_data = df['interest'].value_counts()
            fig = px.bar(x=interest_data.index, y=interest_data.values, 
                        color=interest_data.values, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Budget Distribution")
            fig = px.histogram(df, x='budget_thousand_inr', nbins=30, 
                             color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Budget vs Rating")
            fig = px.scatter(df, x='budget_thousand_inr', y='rating_given', 
                           opacity=0.5, color='place_type')
            st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🌟 AI-Powered Travel Recommendation System
    
    This is an end-to-end machine learning project that combines:
    
    ### 🎯 Features:
    - **Smart Recommendations**: ML models trained on 20,000+ data points
    - **AI Chatbot**: Powered by Groq's Llama 3.3 (70B parameters)
    - **Interactive UI**: Built with Streamlit
    - **Advanced Analytics**: Comprehensive travel insights
    - **Real-time Predictions**: Instant destination matching
    
    ### 🛠️ Tech Stack:
    - **ML/DL**: Scikit-learn, XGBoost, TensorFlow
    - **Generative AI**: Groq API (Llama 3.3)
    - **Visualization**: Plotly, Seaborn, Matplotlib
    - **UI**: Streamlit
    - **Deployment**: Ready for GitHub, Streamlit Cloud
    
    ### 📚 Models Used:
    - Random Forest
    - XGBoost
    - Gradient Boosting
    - Deep Neural Network
    - And more...
    
    ### 👨‍💻 Developer:
    Built with ❤️ using AI assistance
    
    ### 🚀 Get Started:
    1. Clone repository from GitHub
    2. Install dependencies: `pip install -r requirements.txt`
    3. Set GROQ_API_KEY environment variable
    4. Run: `streamlit run app.py`
    
    ---
    
    Made with 🤖 AI + 👨‍💻 Code + ❤️ Passion
    """)

if __name__ == "__main__":
    main()