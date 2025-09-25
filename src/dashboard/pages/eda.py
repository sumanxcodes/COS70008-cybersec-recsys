# src/pages/eda.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

st.title("EDA")
st.caption("Exploratory Data Analysis")

# ── paths ─────────────────────────────────────────────────────────────────────
def find_src_dir(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "src":
            return p
    return start

HERE = Path(__file__).resolve()
SRC_DIR = find_src_dir(HERE)
DATA_DIR = SRC_DIR / "data"

FILES = {
    "orgs": DATA_DIR / "orgs_full.csv",
    "exs":  DATA_DIR / "exercises_full.csv",
}

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_read(path: Path):
    if not path.exists():
        st.error(f"Missing file: `{path}`")
        return None
    try:
        return load_csv(path)
    except Exception as e:
        st.error(f"Failed to load `{path}`\n{e}")
        return None

orgs = safe_read(FILES["orgs"])
exs  = safe_read(FILES["exs"])

# ── utils ─────────────────────────────────────────────────────────────────────
def split_semi(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
        .replace({"": None})
        .dropna()
    )

def normalize_explode(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    s = split_semi(df[col]) if col in df.columns else pd.Series(dtype=str)
    vc = s.value_counts().reset_index()
    vc.columns = [col, "count"]
    return vc.head(top_n)

def numeric_hist(df: pd.DataFrame, col: str, bins: int = 20):
    if col not in df.columns:
        return None
    data = df[[col]].dropna()
    if data.empty:
        return None
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
            y=alt.Y("count()", title="count"),
            tooltip=[alt.Tooltip(f"{col}:Q", format=".2f"), alt.Tooltip("count()", title="count")],
        )
        .properties(height=200)
    )

def heatmap(df: pd.DataFrame, cols: list[str]):
    sub = df[cols].dropna()
    if sub.empty or sub.shape[1] < 2:
        return None
    corr = sub.corr(numeric_only=True)
    corr_df = corr.reset_index().melt("index")
    corr_df.columns = ["x", "y", "corr"]
    return (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("x:N", title=None),
            y=alt.Y("y:N", title=None),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange"), title="correlation"),
            tooltip=["x:N", "y:N", alt.Tooltip("corr:Q", format=".2f")],
        )
        .properties(height=280)
    )

# ── token combiner ────────────────────────────────────────────────────────────
def combine_tokens(row: pd.Series, cols=("Threats", "TTPs", "Aims")) -> str:
    parts = []
    for col in cols:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]))
    return ";".join(parts)


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
TOP_N = st.sidebar.slider("Top N for categorical counts", 5, 50, 20, 5)
BINS  = st.sidebar.slider("Histogram bins", 10, 60, 25, 5)
SHOW_FULL_TABLES = st.sidebar.toggle("Show full tables (else head)", value=False)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_exs, tab_orgs, tab_overlap, tab_clusters = st.tabs([
    "📘 Exercises",
    "🏢 Organisations",
    "🔁 Overlap",
    "🌐 Org Clusters"
])

# ================================ EXERCISES ===================================
with tab_exs:
    st.subheader("Exercises: distributions & numerics")
    if exs is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("rows", f"{len(exs):,}")
        c2.metric("cols", f"{exs.shape[1]}")
        c3.caption(f"📂 {FILES['exs']}")

        st.markdown("**Missing values**")
        mv = exs.isna().mean().sort_values(ascending=False).to_frame("missing_%")
        mv["missing_%"] = (mv["missing_%"] * 100).round(1)
        st.dataframe(mv if SHOW_FULL_TABLES else mv.head(20), use_container_width=True)

        cat_cols = ["ExThreat", "ExTTPs", "ExGroups", "ExSoftware", "ExCategories", "ExAudience"]
        cat_cols_present = [c for c in cat_cols if c in exs.columns]

        st.markdown("### Top categorical tokens")
        grids = st.columns(3)
        for i, col in enumerate(cat_cols_present):
            with grids[i % 3]:
                st.markdown(f"**Top {TOP_N} — {col}**")
                top = normalize_explode(exs, col, TOP_N)
                st.dataframe(top, use_container_width=True, height=300)
                if not top.empty:
                    chart = (
                        alt.Chart(top)
                        .mark_bar()
                        .encode(
                            y=alt.Y(f"{col}:N", sort="-x"),
                            x=alt.X("count:Q"),
                            tooltip=[col, "count"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)

        num_cols = [c for c in ["ExMaturity", "ExComplexity", "ExLength", "ExTradeCraftIntra", "ExTradeCraftInter"] if c in exs.columns]
        if num_cols:
            st.markdown("### Numeric summaries")
            st.dataframe(exs[num_cols].describe().T.assign(missing=exs[num_cols].isna().sum()), use_container_width=True)

            st.markdown("### Histograms")
            grid = st.columns(3)
            for i, col in enumerate(num_cols):
                chart = numeric_hist(exs, col, bins=BINS)
                if chart is not None:
                    with grid[i % 3]:
                        st.altair_chart(chart, use_container_width=True)

            st.markdown("### Correlation (numeric)")
            hm = heatmap(exs, num_cols)
            if hm is not None:
                st.altair_chart(hm, use_container_width=True)
    else:
        st.info("Exercises dataset not available.")

# =============================== ORGANISATIONS ================================
with tab_orgs:
    st.subheader("Organisations: distributions")
    if orgs is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("rows", f"{len(orgs):,}")
        c2.metric("cols", f"{orgs.shape[1]}")
        c3.caption(f"📂 {FILES['orgs']}")

        st.markdown("**Missing values**")
        mv = orgs.isna().mean().sort_values(ascending=False).to_frame("missing_%")
        mv["missing_%"] = (mv["missing_%"] * 100).round(1)
        st.dataframe(mv if SHOW_FULL_TABLES else mv.head(20), use_container_width=True)

        simple_cats = [c for c in ["Industry", "Region", "Size", "PrimarySecurityTeam"] if c in orgs.columns]
        st.markdown("### Category counts")
        grid = st.columns(2)
        for i, col in enumerate(simple_cats):
            with grid[i % 2]:
                st.markdown(f"**Top {TOP_N} — {col}**")
                vc = orgs[col].fillna("∅").value_counts().head(TOP_N).reset_index()
                vc.columns = [col, "count"]
                st.dataframe(vc, use_container_width=True, height=300)
                chart = (
                    alt.Chart(vc)
                    .mark_bar()
                    .encode(
                        y=alt.Y(f"{col}:N", sort="-x"),
                        x=alt.X("count:Q"),
                        tooltip=[col, "count"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)

        token_cols = [c for c in ["Threats", "TTPs", "Aims"] if c in orgs.columns]
        st.markdown("### Tokenized lists")
        grid2 = st.columns(3)
        for i, col in enumerate(token_cols):
            with grid2[i % 3]:
                st.markdown(f"**Top {TOP_N} — {col}**")
                top = normalize_explode(orgs, col, TOP_N)
                st.dataframe(top, use_container_width=True, height=300)
                if not top.empty:
                    chart = (
                        alt.Chart(top)
                        .mark_bar()
                        .encode(
                            y=alt.Y(f"{col}:N", sort="-x"),
                            x=alt.X("count:Q"),
                            tooltip=[col, "count"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Organisations dataset not available.")

# ================================= OVERLAP ===================================
with tab_overlap:
    st.subheader("Org vs Exercise: TTP overlap")
    if orgs is not None and exs is not None and "TTPs" in orgs.columns and "ExTTPs" in exs.columns:
        org_ttp = split_semi(orgs["TTPs"]).value_counts().to_frame("org_count")
        ex_ttp  = split_semi(exs["ExTTPs"]).value_counts().to_frame("ex_count")
        both = org_ttp.join(ex_ttp, how="outer").fillna(0).astype(int)
        both["org_rank"] = both["org_count"].rank(ascending=False, method="dense")
        both["ex_rank"]  = both["ex_count"].rank(ascending=False, method="dense")
        both["gap"] = both["org_rank"] - both["ex_rank"]  # positive = orgs care more than coverage ranks

        st.caption("Higher **gap** suggests demand > coverage.")
        top = both.sort_values(["org_count", "ex_count"], ascending=[False, False]).head(TOP_N)
        st.dataframe(top, use_container_width=True)

        chart = (
            alt.Chart(both.reset_index().rename(columns={"index": "TTP"}))
            .mark_circle()
            .encode(
                x=alt.X("org_count:Q", title="Org demand (count)"),
                y=alt.Y("ex_count:Q", title="Exercise coverage (count)"),
                size=alt.Size("gap:Q", title="gap (org_rank - ex_rank)", legend=None),
                tooltip=["TTP:N", "org_count:Q", "ex_count:Q", "gap:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Overlap view needs `orgs.TTPs` and `exs.ExTTPs`.")

# =============================== CLUSTERS (Altair) ============================
with tab_clusters:
    st.subheader("Organisation Clusters — Threat Profiles")

    if orgs is not None:

        # ensure all_tokens exists (and is fresh) from available cols
        required_cols = [c for c in ["Threats", "TTPs", "Aims"] if c in orgs.columns]
        if "all_tokens" not in orgs.columns and required_cols:
            orgs["all_tokens"] = orgs.apply(lambda r: combine_tokens(r, cols=required_cols), axis=1)

        if "all_tokens" in orgs.columns:
            # sidebar controls
            k = st.sidebar.number_input("K (clusters)", min_value=2, max_value=10, value=4, step=1)
            min_df = st.sidebar.slider("Min doc freq (token filter)", 1, 10, 2, 1,
                                       help="Ignore tokens that appear in fewer than this many orgs")

            # keep only orgs that actually have tokens
            work = orgs.copy()
            work["all_tokens"] = work["all_tokens"].fillna("").astype(str).str.strip()
            work = work[work["all_tokens"] != ""]
            if work.empty:
                st.info("No organisations with token data to cluster.")
            else:
                # sparse token matrix (binary presence)
                vec = CountVectorizer(
                    tokenizer=lambda x: [t.strip() for t in x.split(";") if t.strip()],
                    token_pattern=None,
                    binary=True,
                    min_df=min_df
                )
                X = vec.fit_transform(work["all_tokens"])

                # project to 2D without densifying
                svd = TruncatedSVD(n_components=2, random_state=42)
                X_2d = svd.fit_transform(X)
                ev = svd.explained_variance_ratio_.sum()

                # kmeans clustering
                kmeans = KMeans(n_clusters=int(k), random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                work["cluster"] = labels.astype(int)

                # plot dataframe
                emb = pd.DataFrame(X_2d, columns=["comp1", "comp2"], index=work.index)
                emb["cluster"] = work["cluster"].astype(str)
                emb["ORGID"] = work["ORGID"] if "ORGID" in work.columns else np.arange(len(work))

                st.caption(f"SVD 2D projection explains ~{ev*100:.1f}% variance")

                # interactive scatter (legend filter)
                sel = alt.selection_point(fields=["cluster"], bind="legend")
                chart = (
                    alt.Chart(emb)
                    .mark_circle(size=85, opacity=0.85)
                    .encode(
                        x=alt.X("comp1:Q", title="Component 1"),
                        y=alt.Y("comp2:Q", title="Component 2"),
                        color=alt.Color("cluster:N", title="Cluster"),
                        tooltip=["ORGID:N", "cluster:N", alt.Tooltip("comp1:Q", format=".2f"), alt.Tooltip("comp2:Q", format=".2f")],
                        opacity=alt.condition(sel, alt.value(1), alt.value(0.2))
                    )
                    .add_params(sel)
                    .properties(height=420)
                )
                st.altair_chart(chart, use_container_width=True)

                # cluster sizes
                sizes = work["cluster"].value_counts().sort_index()
                st.markdown("### Cluster sizes")
                st.dataframe(sizes.rename("n").to_frame(), use_container_width=True)

                # per-cluster top tokens by document frequency (percentage of orgs in cluster containing the token)
                st.markdown("### Cluster summaries (top tokens)")
                vocab = np.array(vec.get_feature_names_out())

                # precompute boolean presence for doc frequency
                X_bool = (X > 0).astype(int)

                for c in sorted(work["cluster"].unique()):
                    mask = (work["cluster"] == c).values
                    n_c = int(mask.sum())
                    if n_c == 0:
                        continue

                    Xc_bool = X_bool[mask]
                    # document frequency per token inside cluster
                    df_counts = np.asarray(Xc_bool.sum(axis=0)).ravel()
                    if df_counts.sum() == 0:
                        st.markdown(f"#### Cluster {c} (n={n_c})")
                        st.info("No tokens found for this cluster.")
                        continue

                    # top 10 by doc freq
                    top_idx = df_counts.argsort()[::-1][:10]
                    top = pd.DataFrame({
                        "token": vocab[top_idx],
                        "doc_freq": df_counts[top_idx].astype(int),
                    })
                    top["pct_orgs_in_cluster"] = (top["doc_freq"] / n_c * 100).round(1)

                    st.markdown(f"#### Cluster {c} (n={n_c}) — top tokens by document frequency")
                    st.dataframe(top, use_container_width=True, hide_index=True)

                    bar = (
                        alt.Chart(top)
                        .mark_bar()
                        .encode(
                            x=alt.X("pct_orgs_in_cluster:Q", title="% of orgs in cluster"),
                            y=alt.Y("token:N", sort="-x", title=None),
                            tooltip=["token:N", "doc_freq:Q", alt.Tooltip("pct_orgs_in_cluster:Q", format=".1f")]
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(bar, use_container_width=True)

                # optional: write clusters back to main df to use elsewhere
                orgs.loc[work.index, "cluster"] = work["cluster"]
        else:
            st.info("Clusters need `orgs.all_tokens` column.")
    else:
        st.info("Organisations dataset not available.")

st.divider()
st.caption("Use the tabs to switch between Exercises, Organisations, Overlap, and Clusters.")
