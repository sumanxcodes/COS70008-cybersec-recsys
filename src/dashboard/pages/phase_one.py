# pages/phase_one.py
# ──────────────────────────────────────────────────────────────────────────────
# COS70008 — Technology Innovation Research & Project
# Phase 1: Baseline Exercise↔Exercise Recommender (Cosine on Threats + TTPs)
#
# Why this page exists (student-friendly):
# - Pick a seed exercise and show its Top-K most similar neighbours.
# - Let me sanity-check the results visually.
# - Provide a simple, data-only evaluation using metadata (Threats/TTPs).
#
# Notes:
# - No user interaction history yet, so all evaluation is “metadata-based”.
# - Threats/TTPs are multivalue (semicolon separated); we compare as sets.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# ---------- Page header ----------
st.title("Phase 1 — Baseline Exercise Recommender")
st.caption(
    "Cosine similarity over Threats + TTPs (document-term model). "
    "This page is a baseline: simple, explainable, and a reference for later phases."
)

# ---------- Paths ----------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2] if HERE.parent.name == "pages" else HERE.parent
DATA_DIR = SRC_DIR / "data"

SIM_FILE = DATA_DIR / "ex_sim.npz"          # contains S (similarity matrix)
EX_FILE  = DATA_DIR / "exercises_full.csv"  # exercise metadata

# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def load_sim(path: Path) -> np.ndarray:
    npz = np.load(path)
    return npz["S"]  # already scaled to [0, 1]

@st.cache_data(show_spinner=False)
def load_exercises(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# Guard: required files
if not SIM_FILE.exists() or not EX_FILE.exists():
    st.error("Missing files in `src/data/`: expected `ex_sim.npz` and `exercises_full.csv`.")
    st.stop()

# ---------- Data ----------
S = load_sim(SIM_FILE)
df_exs = load_exercises(EX_FILE).reset_index().rename(columns={"index": "idx"})

# Readable dropdown labels
df_exs["Label"] = (
    "EXID " + df_exs["EXID"].astype(str)
    + "  • Threat: " + df_exs["ExThreat"].fillna("")
    + "  • TTPs: " + df_exs["ExTTPs"].fillna("")
)

# ---------- Sidebar (simple + clear) ----------
with st.sidebar:
    st.header("Controls")
    st.write("Pick a seed exercise and how many similar items to show.")
    seed_label = st.selectbox("Seed exercise", df_exs["Label"])
    K = st.slider("Top-K similar", 1, 20, 5)
    show_network = st.checkbox("Show radial network", value=True)
    st.caption("Tip: use K≈5 for quick checks; increase K to inspect coverage/diversity.")

# Resolve the selected seed
seed_idx = int(df_exs.loc[df_exs["Label"] == seed_label, "idx"].iloc[0])
seed_id  = int(df_exs.loc[df_exs["Label"] == seed_label, "EXID"].iloc[0])

# Similarities for the seed
sim_row = S[seed_idx].astype(float).copy()
sim_row[seed_idx] = -1  # exclude self
top_idx = np.argsort(-sim_row)[:K]

# ---------- Tabs ----------
recs_tab, eval_tab = st.tabs(["Recommendations", "Evaluation"])

# =============================================================================
# TAB 1 — Recommendations (what the system suggests)
# =============================================================================
with recs_tab:
    # Seed row (for easy comparison)
    seed_row = df_exs.loc[df_exs["idx"] == seed_idx, [
        "EXID", "ExThreat", "ExTTPs", "ExGroups", "ExComplexity", "ExMaturity"
    ]].copy()
    seed_row["Similarity"] = 1.000

    # Top-K neighbours (excluding the seed)
    recs = df_exs.iloc[top_idx][[
        "EXID", "ExThreat", "ExTTPs", "ExGroups", "ExComplexity", "ExMaturity"
    ]].copy()
    recs["Similarity"] = sim_row[top_idx].round(3)

    # Merge for display
    display_df = pd.concat(
        [seed_row.assign(Role="Seed"),
         recs.assign(Role="Recommended")],
        ignore_index=True
    )[
        ["Role", "EXID", "ExThreat", "ExTTPs", "ExGroups", "ExComplexity", "ExMaturity", "Similarity"]
    ]

    # Highlight the seed row (robust across Streamlit versions)
    def _highlight_seed(row):
        return ["background-color: #e6ffe6; font-weight: 600;" if row["Role"] == "Seed" else ""] * len(row)

    styled = display_df.style.apply(_highlight_seed, axis=1)
    try:
        html = styled.hide(axis="index").to_html()
        st.subheader(f"Top-{K} similar to EXID {seed_id}")
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.subheader(f"Top-{K} similar to EXID {seed_id}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("I expect similar Threats/TTPs and comparable complexity/maturity.")

    # ---- Visual 1: similarity bars (Top-K) ----
    plot_df = recs.sort_values("Similarity", ascending=True).copy()
    plot_df["EXID"] = plot_df["EXID"].astype(str)

    bar = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Similarity:Q", title="Cosine similarity (0–1)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("EXID:N", sort=None, title="EXID"),
            tooltip=[
                alt.Tooltip("EXID:N", title="EXID"),
                alt.Tooltip("Similarity:Q", format=".3f"),
                alt.Tooltip("ExThreat:N", title="Threat(s)"),
                alt.Tooltip("ExTTPs:N", title="TTPs"),
            ],
        )
        .properties(height=320, title=f"Similarity scores for EXID {seed_id}")
    )

    labels = (
        alt.Chart(plot_df)
        .mark_text(align="left", baseline="middle", dx=4)
        .encode(x="Similarity:Q", y=alt.Y("EXID:N", sort=None), text=alt.Text("Similarity:Q", format=".2f"))
    )
    st.altair_chart((bar + labels).interactive(), use_container_width=True)

    # ---- Visual 2: seed-row heatmap (top ~30) ----
    max_cols = 30
    order_n = min(len(sim_row) - 1, max(max_cols, K))
    order_idx = np.argsort(-sim_row)[:order_n]
    hm_df = pd.DataFrame({
        "EXID": df_exs.iloc[order_idx]["EXID"].astype(str).values,
        "Similarity": sim_row[order_idx].astype(float),
    })
    hm_df = hm_df[hm_df["Similarity"] >= 0].copy()
    hm_df["row"] = f"EXID {seed_id}"
    cat_order = hm_df.sort_values("Similarity", ascending=False)["EXID"].tolist()

    heat = (
        alt.Chart(hm_df)
        .mark_rect(stroke="#2a2a2a", strokeWidth=0.5)
        .encode(
            x=alt.X("EXID:N", sort=cat_order, title="Most similar (sorted)"),
            y=alt.Y("row:N", title="", axis=None),
            color=alt.Color("Similarity:Q", title="Similarity", scale=alt.Scale(scheme="yellowgreenblue", domain=[0, 1])),
            tooltip=[alt.Tooltip("EXID:N"), alt.Tooltip("Similarity:Q", format=".3f")],
        )
        .properties(width=alt.Step(24), height=400, title="Seed similarity row")
        .configure_axis(grid=False, domain=False, ticks=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(heat, use_container_width=True)

    # ---- Visual 3: quick distributions (Top-K metadata) ----
    c1, c2 = st.columns(2)
    with c1:
        comp_df = recs[["ExComplexity"]].dropna().astype(str)
        comp_counts = comp_df.value_counts().reset_index(name="count")
        st.altair_chart(
            alt.Chart(comp_counts).mark_bar().encode(
                x=alt.X("ExComplexity:N", sort="ascending", title="ExComplexity"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["ExComplexity:N", "count:Q"]
            ).properties(height=300, title="Top-K: Complexity distribution"),
            use_container_width=True
        )
    with c2:
        mat_df = recs[["ExMaturity"]].dropna().astype(str)
        mat_counts = mat_df.value_counts().reset_index(name="count")
        st.altair_chart(
            alt.Chart(mat_counts).mark_bar().encode(
                x=alt.X("ExMaturity:N", sort="ascending", title="ExMaturity"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["ExMaturity:N", "count:Q"]
            ).properties(height=300, title="Top-K: Maturity distribution"),
            use_container_width=True
        )

    # ---- Visual 4: simple radial network (optional) ----
    def radial_layout(seed_label: str, others: pd.DataFrame):
        nodes = [{"id": seed_label, "x": 0.0, "y": 0.0, "is_seed": True, "Similarity": 1.0}]
        n = len(others)
        for i, row in enumerate(others.itertuples(index=False)):
            theta = 2 * np.pi * i / max(n, 1)
            nodes.append({
                "id": f"EXID {int(row.EXID)}",
                "x": float(np.cos(theta)), "y": float(np.sin(theta)),
                "is_seed": False, "Similarity": float(row.Similarity),
            })
        return pd.DataFrame(nodes), pd.DataFrame({
            "source": seed_label,
            "target": [f"EXID {int(x)}" for x in others["EXID"]],
            "weight": others["Similarity"].astype(float).values
        })

    if show_network:
        seed_node = f"EXID {seed_id}"
        nodes_df, edges_df = radial_layout(seed_node, recs[["EXID", "Similarity"]])
        nodes_df["type"] = np.where(nodes_df["is_seed"], "Seed", "Recommended")

        edges_join = (
            edges_df.merge(nodes_df.add_prefix("src_"), left_on="source", right_on="src_id")
                    .merge(nodes_df.add_prefix("dst_"), left_on="target", right_on="dst_id")
        )

        edges_chart = (
            alt.Chart(edges_join)
            .mark_line(color="#cccccc")
            .encode(
                x=alt.X("src_x:Q", axis=None), y=alt.Y("src_y:Q", axis=None),
                x2="dst_x:Q", y2="dst_y:Q",
                strokeWidth=alt.StrokeWidth("weight:Q", scale=alt.Scale(range=[1, 6]), legend=None),
                tooltip=[alt.Tooltip("source:N", title="Seed"),
                         alt.Tooltip("target:N", title="Recommended"),
                         alt.Tooltip("weight:Q", format=".3f", title="Similarity")],
            )
        )

        nodes_chart = (
            alt.Chart(nodes_df)
            .mark_circle(opacity=1.0)
            .encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                size=alt.Size("Similarity:Q", scale=alt.Scale(range=[900, 3600]), legend=None),
                color=alt.Color("type:N", title="Node type",
                                scale=alt.Scale(domain=["Seed", "Recommended"],
                                                range=["#1f77b4", "#bdbdbd"])),
                tooltip=[alt.Tooltip("id:N", title="Exercise"),
                         alt.Tooltip("Similarity:Q", format=".3f")],
            )
        )

        labels_chart = (
            alt.Chart(nodes_df)
            .mark_text(baseline="middle", fontSize=11, fontWeight="bold", color="black")
            .encode(x="x:Q", y="y:Q", text="id:N")
        )

        network = (edges_chart + nodes_chart + labels_chart).properties(
            width="container", height=480, title="Similarity network (Seed → Top-K)"
        ).configure_axis(grid=False, domain=False, ticks=False, labels=False)

        st.altair_chart(network, use_container_width=True)

# =============================================================================
# TAB 2 — Evaluation (metadata-based, dynamic)
# Rationale (short and honest):
# - No train/test interactions yet, so I check relevance using the metadata.
# - Threats/TTPs are multivalue; I split on ';' and compare as sets.
# - Metrics shown: Hit-Rate@K (any Threat overlap), Threat-overlap@K, TTP-overlap@K.
# =============================================================================
with eval_tab:
    st.header("Phase-1 Evaluation (metadata-based)")
    st.caption(
        "I treat two items as related if they share labels. "
        "Hit-Rate means: in the Top-K, did we get at least one matching Threat?"
    )

    # ---------- helpers for multivalue cells ----------
    def _tok_norm(t) -> str:
        return str(t).strip().casefold()

    def _iter_tokens(x):
        """Yield tokens whether cell is list-like or 'a;b;c' or NaN."""
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return
        if isinstance(x, (list, tuple, set)):
            for t in x:
                tt = _tok_norm(t)
                if tt:
                    yield tt
        else:
            for t in str(x).split(";"):
                tt = _tok_norm(t)
                if tt and tt != "nan":
                    yield tt

    def _to_set(x) -> set:
        return set(_iter_tokens(x))

    def _topk_indices_for_seed(idx: int, k: int) -> np.ndarray:
        sr = S[idx].astype(float).copy()
        sr[idx] = -1
        k = max(0, min(k, sr.shape[0] - 1))
        return np.argsort(-sr)[:k]

    # ---------- metric functions ----------
    def hit_rate_any_threat_overlap(seed_idx_local: int, top_idx_local: np.ndarray) -> float:
        seed_threats = _to_set(df_exs.loc[seed_idx_local, "ExThreat"])
        if not seed_threats or len(top_idx_local) == 0:
            return 0.0
        rec_threats = df_exs.iloc[top_idx_local]["ExThreat"].tolist()
        return float(any(_to_set(s) & seed_threats for s in rec_threats))  # 1 or 0

    def threat_overlap_at_k(seed_idx_local: int, top_idx_local: np.ndarray) -> float:
        seed_threats = _to_set(df_exs.loc[seed_idx_local, "ExThreat"])
        if not seed_threats or len(top_idx_local) == 0:
            return 0.0
        overlaps = []
        for s in df_exs.iloc[top_idx_local]["ExThreat"]:
            rec = _to_set(s)
            overlaps.append(len(seed_threats & rec) / len(seed_threats))
        return float(np.mean(overlaps)) if overlaps else 0.0

    def ttp_overlap_at_k(seed_idx_local: int, top_idx_local: np.ndarray) -> float:
        seed_ttps = _to_set(df_exs.loc[seed_idx_local, "ExTTPs"])
        if not seed_ttps or len(top_idx_local) == 0:
            return 0.0
        overlaps = []
        for s in df_exs.iloc[top_idx_local]["ExTTPs"]:
            rec = _to_set(s)
            overlaps.append(len(seed_ttps & rec) / len(seed_ttps))
        return float(np.mean(overlaps)) if overlaps else 0.0

    # ---------- compute metrics for current seed ----------
    top_idx_eval = _topk_indices_for_seed(seed_idx, K)
    hr_seed  = hit_rate_any_threat_overlap(seed_idx, top_idx_eval)
    thr_ovl  = threat_overlap_at_k(seed_idx, top_idx_eval)
    ttp_ovl  = ttp_overlap_at_k(seed_idx, top_idx_eval)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Hit-Rate@{K} (any Threat overlap) — EXID {seed_id}", f"{hr_seed:.2f}")
    with c2:
        st.metric(f"Threat-overlap@{K} — EXID {seed_id}", f"{thr_ovl:.2f}")
    with c3:
        st.metric(f"TTP-overlap@{K} — EXID {seed_id}", f"{ttp_ovl:.2f}")

    # ---- Visual A: small bar for the seed ----
    seed_metrics_df = pd.DataFrame({
        "Metric": [f"Hit@{K}", f"Threat@{K}", f"TTP@{K}"],
        "Value":  [hr_seed,    thr_ovl,       ttp_ovl]
    })
    st.altair_chart(
        alt.Chart(seed_metrics_df).mark_bar().encode(
            x=alt.X("Metric:N", title=""),
            y=alt.Y("Value:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Metric:N", "Value:Q"]
        ).properties(title=f"Seed EXID {seed_id} — metrics at K={K}", height=220)
        + alt.Chart(seed_metrics_df).mark_text(dy=-6).encode(
            x="Metric:N", y="Value:Q", text=alt.Text("Value:Q", format=".2f")
        ),
        use_container_width=True
    )

    # ---- Visual B: curves vs K for the seed ----
    max_k = min(20, S.shape[0] - 1)
    rows = []
    for k in range(1, max_k + 1):
        ti = _topk_indices_for_seed(seed_idx, k)
        rows.append({
            "K": k,
            "Hit":    hit_rate_any_threat_overlap(seed_idx, ti),
            "Threat": threat_overlap_at_k(seed_idx, ti),
            "TTP":    ttp_overlap_at_k(seed_idx, ti),
        })
    kseed_df = pd.DataFrame(rows).melt("K", var_name="Metric", value_name="Score")
    st.altair_chart(
        alt.Chart(kseed_df).mark_line(point=True).encode(
            x=alt.X("K:Q", title="K"),
            y=alt.Y("Score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N"),
            tooltip=["K:Q", "Metric:N", "Score:Q"]
        ).properties(title="Seed metric curves vs K", height=260),
        use_container_width=True
    )

    # ---- Optional: corpus averages + coverage/diversity ----
    show_corpus = st.checkbox("Across all exercises (averages + coverage)")
    if show_corpus:
        hrs, thr_ovls, ttp_ovls, all_recs = [], [], [], set()
        for i in range(len(df_exs)):
            ti = _topk_indices_for_seed(i, K)
            hrs.append(hit_rate_any_threat_overlap(i, ti))
            thr_ovls.append(threat_overlap_at_k(i, ti))
            ttp_ovls.append(ttp_overlap_at_k(i, ti))
            all_recs.update(df_exs.iloc[ti]["EXID"].tolist())

        st.write(f"**Avg Hit-Rate@{K}:** {np.mean(hrs):.2f}")
        st.write(f"**Avg Threat-overlap@{K}:** {np.mean(thr_ovls):.2f}")
        st.write(f"**Avg TTP-overlap@{K}:** {np.mean(ttp_ovls):.2f}")

        coverage = len(all_recs) / len(df_exs)
        st.write(f"**Coverage@{K}:** {coverage:.2f}")

        # Diversity = 1 − average similarity among Top-K set (higher = more varied)
        try:
            diversities = []
            for i in range(len(df_exs)):
                ti = _topk_indices_for_seed(i, K)
                if len(ti) > 1:
                    diversities.append(1.0 - float(np.mean(S[ti][:, ti].astype(float))))
            if diversities:
                st.write(f"**Avg Diversity@{K}:** {np.mean(diversities):.2f}")
        except Exception:
            pass

        # Histograms (distributions)
        dist_df = pd.DataFrame({"Hit": hrs, "Threat": thr_ovls, "TTP": ttp_ovls}).melt(
            var_name="Metric", value_name="Score"
        )
        st.altair_chart(
            alt.Chart(dist_df).mark_bar().encode(
                x=alt.X("Score:Q", bin=alt.Bin(maxbins=20), scale=alt.Scale(domain=[0, 1]),
                        title="Score (0–1)"),
                y=alt.Y("count():Q", title="Count of exercises"),
                color="Metric:N",
                tooltip=["Metric:N", "count():Q"]
            ).properties(title=f"Corpus distributions at K={K}", height=260),
            use_container_width=True
        )

        # Coverage vs K (quick sweep)
        if st.checkbox("Coverage vs K (corpus sweep)"):
            rows = []
            for k in range(1, max_k + 1):
                all_recs_k = set()
                for i in range(len(df_exs)):
                    ti = _topk_indices_for_seed(i, k)
                    all_recs_k.update(df_exs.iloc[ti]["EXID"].tolist())
                rows.append({"K": k, "Coverage": len(all_recs_k) / len(df_exs)})
            cov_df = pd.DataFrame(rows)
            st.altair_chart(
                alt.Chart(cov_df).mark_line(point=True).encode(
                    x=alt.X("K:Q", title="K"),
                    y=alt.Y("Coverage:Q", title="Coverage", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["K:Q", "Coverage:Q"]
                ).properties(title="Coverage vs K", height=220),
                use_container_width=True
            )

# ---------- Footer ----------
st.divider()
st.caption(
    "Summary: this is a clean baseline. It clusters by Threat reasonably well and "
    "partly by TTPs. Later phases (CF/Hybrid) should lift overlap scores with real usage data."
)
