import streamlit as st

st.set_page_config(
    page_title="Cyber Exercise Recommender",
    page_icon=":material/security:",
    layout="wide"
)

# Pages via the new navigation API (Streamlit >= 1.37)
project_brief = st.Page("pages/project_brief.py", title="Project Brief", icon=":material/article:")
dataset       = st.Page("pages/dataset.py",      title="Dataset",       icon=":material/database:")
eda          = st.Page("pages/eda.py",      title="EDA",       icon=":material/analytics:")
phase_one     = st.Page("pages/phase_one.py",    title="Phase 1",       icon=":material/route:")
phase_two     = st.Page("pages/phase_two.py",    title="Phase 2",       icon=":material/bolt:")
phase_three   = st.Page("pages/phase_three.py",  title="Phase 3",       icon=":material/network_intel_node:")

nav = st.navigation({
    "Overview": [project_brief],
    "Work": [dataset, eda, phase_one, phase_two, phase_three],
})

nav.run()
