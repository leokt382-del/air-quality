
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Air Quality & Health Impact",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Air Quality & Health Impact Analysis")

st.markdown(
    """
    ## Welcome 👋

    This application analyzes air quality data and predicts
    the potential health impact using machine learning.
    """
)

# Dashboard cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset Records", "5,811")

with col2:
    st.metric("Features", "15")

with col3:
    st.metric("Prediction Model", "Random Forest")

with col4:
    st.metric("Target", "Health Impact Score")

st.divider()

# Navigation information
st.subheader("📌 Application Sections")

st.write("""
Use the sidebar to navigate between:

- 🏠 Dashboard
- 📊 Data Analysis
- 📈 Model Performance
- 🤖 Health Impact Prediction
""")
