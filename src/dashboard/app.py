# src/dashboard/app.py
import streamlit as st

st.set_page_config(page_title="Cybersecurity Exercise Recommender", layout="wide")

st.title("🔐 Cybersecurity Exercise Recommendation System")
st.write("Welcome to our COS70008 group project dashboard! 🎓")

st.header("Phase 1: Baseline Recommender")
st.write("This section will show item-based recommendations based on exercise metadata.")

st.header("Phase 2: Hybrid Model")
st.write("This section will combine collaborative filtering with content-based recommendations.")

st.header("Phase 3: Analytics & Visualisation")
st.write("Here, you'll see performance trends, APT mappings, and feedback insights.")

# Temporary placeholder example
if st.button("Run Sample Recommendation"):
    st.success("Sample recommendation generated!")

st.sidebar.header("Navigation")
st.sidebar.write("Use this panel to filter or select different datasets/models.")
