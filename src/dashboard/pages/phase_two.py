import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Phase 2 — Hybrid / Collaborative")

st.markdown("""
Integrate **historical outcomes** to learn which exercises work for which org archetypes.
""")

st.subheader("Data Needed")
st.markdown("""
- `ratings_train_full.csv` with **ORGID**, **EXID**, **ExerciseResults**, **ExerciseRating (1–5)**  
- `orgs_full.csv` for context features (threats, TTPs, aims, maturity).  
- `exercises_full.csv` for item features (threats, TTPs, maturity, tradecraft).
""")

# Uploaders
rats = st.file_uploader("Upload ratings_train_full.csv", type=["csv"], key="rats_p2")
orgs = st.file_uploader("Upload orgs_full.csv", type=["csv"], key="orgs_p2")
exs  = st.file_uploader("Upload exercises_full.csv", type=["csv"], key="exs_p2")

if rats:
    df_r = pd.read_csv(rats)
    st.markdown("**Ratings sample**")
    st.dataframe(df_r.head())

st.subheader("Hybrid Plan")
with st.expander("A) Collaborative Filtering (explicit or implicit)"):
    st.markdown("""
    - **User** = ORGID, **Item** = EXID, **Rating** = ExerciseRating or derived from ExerciseResults.  
    - Try matrix factorisation (ALS/SGD) or LightFM.  
    - Generate CF scores for (ORGID, EXID).
    """)

with st.expander("B) Content + CF Fusion"):
    st.markdown("""
    - From Phase 1, you already have **content similarity** scores.  
    - Combine with CF via:  
      - **Weighted sum**: `score = α·CF + (1−α)·Content`  
      - **Learning to Rank**: train a model on features (CF score, similarity, TTP overlap, maturity match, etc.) to predict click/rating.
    """)

with st.expander("C) Evaluation Metrics"):
    st.markdown("""
    - **Ranking**: precision@k, recall@k, MAP, nDCG.  
    - **Diversity/Coverage**: catalog coverage, intra-list diversity.  
    - **Context fit**: threat/TTP alignment rate; maturity/complexity mismatch penalties.  
    - **Ablations**: CF only vs Content only vs Fusion.
    """)

st.subheader("Outputs")
st.markdown("""
- Ranked recommendations per org with **explanations** (matched TTPs, expected complexity fit).  
- Dashboard panels for **what-if** tuning (α slider, filter by threat family, exclude over-repeated drills).
""")
