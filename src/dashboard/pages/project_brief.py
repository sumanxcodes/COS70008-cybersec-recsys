import streamlit as st

st.title("Project Brief")

st.subheader("Project 1 — Cybersecurity Exercise Recommendation System")
st.markdown("""
Organizations run cybersecurity exercises to assess defences. The goal is to **recommend tailored next-step exercises** based on past performance and the APT landscape.
""")

st.subheader("Phases & Deliverables")
with st.expander("Phase 1 — Baseline Recommender"):
    st.markdown("""
- Build an **item-based** recommender:
  - Extract exercise metadata (ATT&CK technique tags, difficulty, scope).
  - Create a **document-term matrix** (TF-IDF or binary).
  - Compute **pairwise similarities** between exercises.
- **Deliverables:** parameterized prototype + documentation of preprocessing, feature selection, similarity metric.
""")

with st.expander("Phase 2 — Hybrid/Collaborative"):
    st.markdown("""
- Implement collaborative filtering (matrix factorization / LightFM style).
- Fuse CF with content scores (**weighted** or **learning-to-rank**).
- **Deliverables:** trained hybrid model + evaluation against baseline (precision/recall/nDCG, coverage).
""")

with st.expander("Phase 3 — Analytics Dashboard"):
    st.markdown("""
- Interactive web dashboard showing:
  - Historical performance trends
  - APT risk overview
  - Recommended drills with explanations
  - User feedback capture
- **Deliverables:** deployed dashboard + report (architecture, evaluation, lessons learned).
""")

st.subheader("Week-by-Week Plan (Condensed)")
st.markdown("""
- **W1:** Kickoff, ATT&CK & APT review, literature review, roles/tools  
- **W2:** Data exploration & preprocessing; unify schemas  
- **W3:** DTM & feature engineering; vector validation  
- **W4:** Similarity computation; top-N logic  
- **W5:** Baseline testing & evaluation  
- **W6–7:** CF design & implementation; train/evaluate  
- **W8–9:** Hybrid integration; tuning; compare to baseline  
- **W10–11:** Dashboard design & development; integrate engine  
- **W12:** E2E testing; feedback; final report
""")

st.subheader("Datasets (as referenced in brief)")
st.markdown("""
- **orgs_full.csv** — organisation context  
- **exercises_full.csv** — exercise metadata  
- **ratings_train_full.csv** — org×exercise outcomes/ratings
""")
