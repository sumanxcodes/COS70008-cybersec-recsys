# pages/phase_one.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from math import cos, sin, pi

st.title("Phase 1 — Baseline Exercise x Exercise Recommender")
st.caption("Item-to-item similarity using cosine on Threats + TTPs (DTM).")

# ---- paths ----
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2] if HERE.parent.name == "pages" else HERE.parent
DATA_DIR = SRC_DIR / "data"

SIM_FILE = DATA_DIR / "ex_sim.npz"
EX_FILE  = DATA_DIR / "exercises_full.csv"

# ---- load ----
@st.cache_data(show_spinner=False)
def load_sim(path: Path):
    npz = np.load(path)
    return npz["S"]  # already scaled 0–1

@st.cache_data(show_spinner=False)
def load_exercises(path: Path):
    return pd.read_csv(path)

if not SIM_FILE.exists() or not EX_FILE.exists():
    st.error("Missing files in `src/data/`: ex_sim.npz or exercises_full.csv")
    st.stop()

S = load_sim(SIM_FILE)
df_exs = load_exercises(EX_FILE).reset_index().rename(columns={"index": "idx"})

# nice dropdown label
df_exs["Label"] = (
    "EXID " + df_exs["EXID"].astype(str)
    + "  • Threat: " + df_exs["ExThreat"].fillna("")
    + "  • TTPs: " + df_exs["ExTTPs"].fillna("")
)

# ---- sidebar ----
with st.sidebar:
    st.header("Controls")
    seed_label = st.selectbox("Seed exercise", df_exs["Label"])
    K = st.slider("Top-K similar", 1, 20, 5)
    show_network = st.checkbox("Show similarity network (Altair radial)", value=True)

seed_idx = int(df_exs.loc[df_exs["Label"] == seed_label, "idx"].iloc[0])
seed_id  = int(df_exs.loc[df_exs["Label"] == seed_label, "EXID"].iloc[0])

# ---- compute top-K ----
sim_row = S[seed_idx].astype(float).copy()
sim_row[seed_idx] = -1  # exclude self
top_idx = np.argsort(-sim_row)[:K]

recs = df_exs.iloc[top_idx][
    ["EXID", "ExThreat", "ExTTPs", "ExGroups", "ExComplexity", "ExMaturity"]
].copy()
recs["Similarity"] = sim_row[top_idx].round(3)

st.subheader(f"Top-{K} exercises similar to EXID {seed_id}")
st.dataframe(recs, use_container_width=True)

# -------------------- Altair plots --------------------

# 1) Bar chart of similarity scores
plot_df = recs.sort_values("Similarity", ascending=True).copy()
plot_df["EXID"] = plot_df["EXID"].astype(str)

bar = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        x=alt.X("Similarity:Q", title="Cosine similarity (0–1)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("EXID:N", sort=None, title="Exercise ID"),
        tooltip=[
            alt.Tooltip("EXID:N", title="EXID"),
            alt.Tooltip("Similarity:Q", format=".3f"),
            alt.Tooltip("ExThreat:N", title="Threat"),
            alt.Tooltip("ExTTPs:N", title="TTPs")
        ],
    )
    .properties(height=320, title=f"Similarity scores for EXID {seed_id}")
)

labels = (
    alt.Chart(plot_df)
    .mark_text(align="left", baseline="middle", dx=4)
    .encode(
        x="Similarity:Q",
        y=alt.Y("EXID:N", sort=None),
        text=alt.Text("Similarity:Q", format=".2f")
    )
)

st.altair_chart((bar + labels).interactive(), use_container_width=True)

# 2) Seed similarity row heatmap (FIXED: explicit sort by Similarity desc; stable labels)
max_cols = 30  # show up to 30 most similar
order_n = min(len(sim_row) - 1, max(max_cols, K))
order_idx = np.argsort(-sim_row)[:order_n]

hm_df = pd.DataFrame({
    "EXID": df_exs.iloc[order_idx]["EXID"].astype(str).values,
    "Similarity": sim_row[order_idx].astype(float),
})
hm_df = hm_df[hm_df["Similarity"] >= 0].copy()
hm_df["row"] = f"EXID {seed_id}"

# explicit x order: most similar → least
cat_order = hm_df.sort_values("Similarity", ascending=False)["EXID"].tolist()
cmin = float(hm_df["Similarity"].min()) if not hm_df.empty else 0.0
cmax = float(hm_df["Similarity"].max()) if not hm_df.empty else 1.0

heat = (
    alt.Chart(hm_df)
    .mark_rect(stroke="#2a2a2a", strokeWidth=0.5)
    .encode(
        x=alt.X(
            "EXID:N",
            sort=cat_order,
            title="Most similar exercises (sorted by similarity)",
            axis=alt.Axis(labelAngle=0, labelOverlap=True, labelLimit=0),
            scale=alt.Scale(paddingInner=0, paddingOuter=0),
        ),
        y=alt.Y("row:N", title="", axis=None),
        color=alt.Color(
            "Similarity:Q",
            title="Similarity",
            scale=alt.Scale(scheme="yellowgreenblue", domain=[0, 1]),
        ),
        tooltip=[alt.Tooltip("EXID:N"), alt.Tooltip("Similarity:Q", format=".3f")],
    )
    # Make tiles readable in the normal container
    .properties(width=alt.Step(24), height=400, title="Seed similarity row")
    .configure_axis(grid=False, domain=False, ticks=False)
    .configure_view(strokeWidth=0)
)

st.altair_chart(heat, use_container_width=True)




# 3) Metadata distributions (Top-K)
c1, c2 = st.columns(2)

with c1:
    comp_df = recs[["ExComplexity"]].dropna().copy()
    comp_df["ExComplexity"] = comp_df["ExComplexity"].astype(str)
    comp_counts = comp_df.value_counts().reset_index(name="count")
    comp_bar = (
        alt.Chart(comp_counts)
        .mark_bar()
        .encode(
            x=alt.X("ExComplexity:N", sort="ascending", title="ExComplexity"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["ExComplexity:N", "count:Q"]
        )
        .properties(height=300, title="Top-K: Complexity distribution")
    )
    st.altair_chart(comp_bar, use_container_width=True)

with c2:
    mat_df = recs[["ExMaturity"]].dropna().copy()
    mat_df["ExMaturity"] = mat_df["ExMaturity"].astype(str)
    mat_counts = mat_df.value_counts().reset_index(name="count")
    mat_bar = (
        alt.Chart(mat_counts)
        .mark_bar()
        .encode(
            x=alt.X("ExMaturity:N", sort="ascending", title="ExMaturity"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["ExMaturity:N", "count:Q"]
        )
        .properties(height=300, title="Top-K: Maturity distribution")
    )
    st.altair_chart(mat_bar, use_container_width=True)


# 4) Similarity "network" in Altair (radial layout with labels)
def radial_layout(seed_label: str, others: pd.DataFrame, radius: float = 1.0,
                  seed_threat: str = None, seed_ttps: str = None):
    nodes = [{
        "id": seed_label, "x": 0.0, "y": 0.0,
        "is_seed": True, "Similarity": 1.0,
        "ExThreat": seed_threat, "ExTTPs": seed_ttps
    }]
    n = len(others)
    for i, row in enumerate(others.itertuples(index=False)):
        theta = 2 * np.pi * i / max(n, 1)
        nodes.append({
            "id": f"EXID {int(row.EXID)}",
            "x": np.cos(theta), "y": np.sin(theta),
            "is_seed": False,
            "Similarity": float(row.Similarity),
            "ExThreat": getattr(row, "ExThreat"),
            "ExTTPs": getattr(row, "ExTTPs"),
        })
    return pd.DataFrame(nodes), pd.DataFrame({
        "source": seed_label,
        "target": [f"EXID {int(x)}" for x in others["EXID"]],
        "weight": others["Similarity"].astype(float).values
    })

# Network chart: cleaner and more interpretable
if show_network:
    seed_node = f"EXID {seed_id}"
    nodes_df, edges_df = radial_layout(
    seed_node,
    recs[["EXID", "Similarity", "ExThreat", "ExTTPs"]],
    radius=1.0,
    seed_threat=df_exs.loc[seed_idx, "ExThreat"],
    seed_ttps=df_exs.loc[seed_idx, "ExTTPs"],
)


    # readable type
    nodes_df = nodes_df.copy()
    nodes_df["type"] = np.where(nodes_df["is_seed"], "Seed", "Recommended")

    edges_join = (
        edges_df.merge(nodes_df.add_prefix("src_"), left_on="source", right_on="src_id")
                .merge(nodes_df.add_prefix("dst_"), left_on="target", right_on="dst_id")
    )

    # edges (light grey, no legend)
    edges_chart = (
        alt.Chart(edges_join)
        .mark_line(color="#cccccc")
        .encode(
            x=alt.X("src_x:Q", axis=None), y=alt.Y("src_y:Q", axis=None),
            x2="dst_x:Q", y2="dst_y:Q",
            strokeWidth=alt.StrokeWidth("weight:Q", scale=alt.Scale(range=[1, 6]), legend=None),
            tooltip=[
                alt.Tooltip("source:N", title="Seed"),
                alt.Tooltip("target:N", title="Recommended"),
                alt.Tooltip("weight:Q", format=".3f", title="Similarity"),
            ],
        )
    )

    # nodes (solid fill, no opacity; only type legend)
    nodes_chart = (
        alt.Chart(nodes_df)
        .mark_circle(opacity=1.0)
        .encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            size=alt.Size("Similarity:Q", scale=alt.Scale(range=[900, 3600]), legend=None),
            color=alt.Color("type:N",
                            title="Node type",
                            scale=alt.Scale(domain=["Seed", "Recommended"],
                                            range=["#1f77b4", "#bdbdbd"])),
            tooltip=[
                alt.Tooltip("id:N", title="Exercise ID"),
                alt.Tooltip("Similarity:Q", format=".3f", title="Similarity score"),
                alt.Tooltip("ExThreat:N", title="Threat"),
                alt.Tooltip("ExTTPs:N", title="TTPs"),
            ],
        )
    )

    # labels inside nodes
    labels_chart = (
        alt.Chart(nodes_df)
        .mark_text(baseline="middle", fontSize=11, fontWeight="bold", color="black")
        .encode(x="x:Q", y="y:Q", text="id:N", tooltip=[])
    )

    network = (edges_chart + nodes_chart + labels_chart).properties(
        width="container", height=480,
        title="Similarity network (Seed → Top-K exercises)"
    ).configure_axis(grid=False, domain=False, ticks=False, labels=False)

    st.altair_chart(network, use_container_width=True)


st.divider()
st.caption("Use the controls to change the seed and K. Heatmap and network update automatically.")
