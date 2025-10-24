# pages/phase_two.py
# =============================================================================
# Phase 2 — Collaborative Filtering (FunkSVD)
# =============================================================================
# We move from metadata similarity (Phase 1) to a learning model.
# The model learns embeddings for organisations and exercises from historical
# ratings and predicts future performance (1–5 scale).
# We evaluate with RMSE and MAE and produce Top-N recommendations.
# =============================================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set
from scipy.sparse import csr_matrix
import altair as alt

# -----------------------------------------------------------------------------
# Streamlit config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Phase 2 — Collaborative Filtering", layout="wide")

# Import local FunkSVD implementation
try:
    from funk_svd import FunkSVD
except Exception:
    st.error("Could not import funk_svd.py. Ensure it is in your project root or on PYTHONPATH.")
    st.stop()

# -----------------------------------------------------------------------------
# Paths and data files
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2] if HERE.parent.name == "pages" else HERE.parent
DATA_DIR = SRC_DIR / "data"

FILES = {
    "train": DATA_DIR / "ratings_train_full.csv",
    "val":   DATA_DIR / "ratings_validation_full.csv",
    "test":  DATA_DIR / "ratings_test_full.csv",
}

# --- Metadata for enrichment ---
ORG_FILE = DATA_DIR / "orgs_full.csv"
EX_FILE  = DATA_DIR / "exercises_full.csv"

st.title("Phase 2 — Collaborative Filtering (FunkSVD)")
st.caption("Learning from org–exercise performance (1–5 ratings). We evaluate with RMSE/MAE and produce Top-N recommendations.")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def build_encoders(train_df: pd.DataFrame):
    u2ix = {u: i for i, u in enumerate(sorted(train_df["ORGID"].unique()))}
    i2ix = {x: j for j, x in enumerate(sorted(train_df["EXID"].unique()))}
    return u2ix, i2ix

@st.cache_data(show_spinner=False)
def map_indices(df: pd.DataFrame, u2ix: Dict[int, int], i2ix: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["u_idx"] = out["ORGID"].map(u2ix)
    out["i_idx"] = out["EXID"].map(i2ix)
    # Use provided 1–5 ratings; clip just in case of rare out-of-range values
    out["rating"] = (
        pd.to_numeric(out["ExerciseRating"], errors="coerce")
        .clip(lower=1.0, upper=5.0)
        .astype(np.float32)
    )
    return out

@st.cache_data(show_spinner=False)
def df_to_csr(frame: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    f = frame.dropna(subset=["u_idx", "i_idx", "rating"]).copy()
    return csr_matrix(
        (f["rating"], (f["u_idx"].astype(int), f["i_idx"].astype(int))),
        shape=(n_users, n_items),
        dtype=np.float32
    )

@st.cache_data(show_spinner=False)
def seen_by_org(train_df: pd.DataFrame) -> Dict[int, Set[int]]:
    return train_df.groupby("ORGID")["EXID"].apply(set).to_dict()

# ---------- metadata enrichment helpers (FIXED) ----------
def _first_match(cols, candidates):
    # cols can be a pandas Index; convert to a plain set/list to avoid hashing issues
    colset = set(list(cols)) if cols is not None else set()
    for c in candidates:
        if c in colset:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_enrichment(org_path: Path, ex_path: Path):
    """Load org & exercise metadata and return trimmed, display-ready frames."""
    org_df = pd.read_csv(org_path) if org_path.exists() else None
    ex_df  = pd.read_csv(ex_path)  if ex_path.exists()  else None

    org_meta = None
    if org_df is not None:
        org_industry = _first_match(org_df.columns, ["Industry", "industry", "Sector", "OrgIndustry", "IndustrySector"])
        org_ttps     = _first_match(org_df.columns, ["TTPs", "Threats", "OrgTTPs", "Org_TTPs"])
        keep = ["ORGID"]
        if org_industry: keep.append(org_industry)
        if org_ttps:     keep.append(org_ttps)
        if len(keep) > 1:
            org_meta = org_df[keep].rename(columns={
                org_industry: "Industry" if org_industry else None,
                org_ttps: "Org_TTPs"     if org_ttps     else None
            })

    ex_meta = None
    if ex_df is not None:
        ex_ttps = _first_match(ex_df.columns, ["ExTTPs", "ExerciseTTPs", "TTPs", "TTP_list", "Exercise_TTPs"])
        keep = ["EXID"]
        if ex_ttps: keep.append(ex_ttps)
        if len(keep) > 1:
            ex_meta = ex_df[keep].rename(columns={ex_ttps: "ExTTPs"})

    return org_meta, ex_meta


# -----------------------------------------------------------------------------
# Data loading sidebar section
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    use_default = st.toggle("Use default CSVs from /data", value=True)

if use_default:
    missing = [p for p in FILES.values() if not p.exists()]
    if missing:
        st.error("Missing files: " + ", ".join(str(m) for m in missing))
        st.stop()
    train_raw = load_csv(FILES["train"])
    val_raw   = load_csv(FILES["val"])
    test_raw  = load_csv(FILES["test"])
else:
    up_train = st.file_uploader("Train CSV", type="csv")
    up_val   = st.file_uploader("Validation CSV", type="csv")
    up_test  = st.file_uploader("Test CSV", type="csv")
    if not (up_train and up_val and up_test):
        st.stop()
    train_raw = pd.read_csv(up_train)
    val_raw   = pd.read_csv(up_val)
    test_raw  = pd.read_csv(up_test)

# Validate required columns and use provided ratings directly
req_cols = {"ORGID", "EXID", "ExerciseRating"}
for name, df in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
    missing = req_cols - set(df.columns)
    if missing:
        st.error(f"{name} CSV is missing columns: {sorted(missing)}")
        st.stop()

# Optional sanity check on rating range
min_r = pd.concat([train_raw["ExerciseRating"], val_raw["ExerciseRating"], test_raw["ExerciseRating"]]).min()
max_r = pd.concat([train_raw["ExerciseRating"], val_raw["ExerciseRating"], test_raw["ExerciseRating"]]).max()
if not (1.0 <= float(min_r) and float(max_r) <= 5.0):
    st.warning("ExerciseRating values are not within 1–5 across the splits. They will be clipped for training.")

train_df = train_raw.copy()
val_df   = val_raw.copy()
test_df  = test_raw.copy()

u2ix, i2ix = build_encoders(train_df)
train_m = map_indices(train_df, u2ix, i2ix)
val_m   = map_indices(val_df,   u2ix, i2ix)
test_m  = map_indices(test_df,  u2ix, i2ix)

R_train = df_to_csr(train_m, len(u2ix), len(i2ix))
R_val   = df_to_csr(val_m,   len(u2ix), len(i2ix))
R_test  = df_to_csr(test_m,  len(u2ix), len(i2ix))

# -----------------------------------------------------------------------------
# Sidebar: Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("FunkSVD Hyperparameters")
    n_factors = st.slider("Latent factors", 8, 256, 64, step=8)
    n_epochs  = st.slider("Epochs", 5, 200, 40, step=5)
    lr        = st.number_input("Learning rate", 1e-5, 1e-1, 5e-3, step=1e-3, format="%.5f")
    reg_bias  = st.number_input("Regularisation (bias)", 0.0, 1.0, 0.01, step=0.01)
    reg_f     = st.number_input("Regularisation (factors)", 0.0, 1.0, 0.05, step=0.01)
    early     = st.toggle("Early stopping", value=True)
    patience  = st.slider("Patience", 1, 10, 3) if early else 0
    eval_every= st.slider("Eval every (epochs)", 1, 10, 1)
    seed      = st.number_input("Random seed", value=42, step=1)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
train_tab, eval_tab, recs_tab = st.tabs(["Train", "Evaluate", "Recommendations"])

# =============================================================================
# TAB 1 — Train
# =============================================================================
with train_tab:
    # High-level counts
    a, b, c = st.columns(3)
    a.metric("Train rows", len(train_df))
    b.metric("Validation rows", len(val_df))
    c.metric("Test rows", len(test_df))
    st.caption("Each row is one org–exercise pair with a provided rating between 1 and 5.")

    st.subheader("Train FunkSVD model")
    st.write("Matrix factorisation learns hidden patterns in how organisations perform across exercises.")

    # Peek at the training data here (inside the first tab)
    with st.expander("Peek training data (first 10 rows)"):
        st.dataframe(train_df.head(10), use_container_width=True)

    go = st.button("Train / Re-train model", type="primary")

    if go:
        with st.spinner("Training model..."):
            model = FunkSVD(
                n_factors=n_factors, n_epochs=n_epochs, lr=lr,
                reg_bias=reg_bias, reg_factors=reg_f,
                early_stopping=early, patience=patience,
                eval_every=eval_every, seed=seed,
                clip=(1.0, 5.0), verbose=True
            )
            # fit returns a list of (epoch, train_rmse, val_rmse)
            history = model.fit(R_train, R_val)

            # store for other tabs
            st.session_state["cf_model"] = model
            st.session_state["train_hist"] = history
            st.success("Model training completed.")

    # Plot training history (robust)
    if "train_hist" in st.session_state:
        hist_raw = st.session_state["train_hist"]
        if isinstance(hist_raw, list) and len(hist_raw) > 0:
            # Convert to DataFrame with tolerant column handling
            if isinstance(hist_raw[0], (list, tuple)) and len(hist_raw[0]) == 3:
                dfh = pd.DataFrame(hist_raw, columns=["epoch", "train_rmse", "val_rmse"])
            elif isinstance(hist_raw[0], (list, tuple)) and len(hist_raw[0]) == 2:
                dfh = pd.DataFrame(hist_raw, columns=["epoch", "train_rmse"])
            elif isinstance(hist_raw[0], (list, tuple)) and len(hist_raw[0]) == 1:
                dfh = pd.DataFrame({"epoch": np.arange(1, len(hist_raw)+1), "train_rmse": [r[0] for r in hist_raw]})
            else:
                # attempt best effort
                dfh = pd.DataFrame(hist_raw)
                if "epoch" not in dfh.columns:
                    dfh.insert(0, "epoch", np.arange(1, len(dfh)+1))

            if "train_rmse" in dfh.columns:
                c1 = (
                    alt.Chart(dfh)
                    .mark_line(point=True)
                    .encode(x="epoch:Q", y="train_rmse:Q", tooltip=list(dfh.columns))
                    .properties(title="Training RMSE")
                )
                st.altair_chart(c1, use_container_width=True)

            if "val_rmse" in dfh.columns:
                c2 = (
                    alt.Chart(dfh)
                    .mark_line(point=True)
                    .encode(x="epoch:Q", y="val_rmse:Q", tooltip=list(dfh.columns))
                    .properties(title="Validation RMSE")
                )
                st.altair_chart(c2, use_container_width=True)
            st.caption("Both training and validation RMSE should decrease and stabilise.")
        else:
            st.info("No valid training history available to plot yet.")

# =============================================================================
# TAB 2 — Evaluate
# =============================================================================
with eval_tab:
    st.subheader("Evaluate model (RMSE / MAE)")
    st.write("Lower is better. RMSE penalises large errors more than MAE.")

    if "cf_model" not in st.session_state:
        st.info("Train a model first in the Train tab.")
    else:
        model: FunkSVD = st.session_state["cf_model"]

        # ---------- build test predictions dataframe ----------
        rows = []
        n = 0
        se = 0.0
        ae = 0.0
        for u, i, r in test_m[["u_idx", "i_idx", "rating"]].itertuples(index=False, name=None):
            if pd.isna(u) or pd.isna(i):
                continue
            u = int(u); i = int(i)
            p = float(model.predict_one(u, i))
            err = r - p
            rows.append({"u_idx": u, "i_idx": i, "actual": float(r), "pred": p,
                         "error": err, "abs_error": abs(err)})
            se += err * err
            ae += abs(err)
            n += 1

        if n == 0:
            st.warning("No test rows matched the train encoders. Check your splits.")
            st.stop()

        rmse = (se / n) ** 0.5
        mae  = (ae / n)

        c1, c2 = st.columns(2)
        c1.metric("Test RMSE", f"{rmse:.3f}")
        c2.metric("Test MAE",  f"{mae:.3f}")
        st.caption("Only counting org/exercise pairs present in the train encoders (no cold-start).")

        preds_df = pd.DataFrame(rows)

        # ---------- 1) Actual vs Predicted scatter ----------
        lo, hi = 1.0, 5.0
        ref_df = pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})

        scatter = (
            alt.Chart(preds_df)
            .mark_circle(size=40, opacity=0.35)
            .encode(
                x=alt.X("actual:Q", title="Actual rating", scale=alt.Scale(domain=[lo, hi])),
                y=alt.Y("pred:Q",   title="Predicted rating", scale=alt.Scale(domain=[lo, hi])),
                tooltip=["u_idx", "i_idx", alt.Tooltip("actual:Q", format=".2f"),
                         alt.Tooltip("pred:Q", format=".2f"), alt.Tooltip("abs_error:Q", format=".2f")]
            )
            .properties(title="Actual vs Predicted (Test)")
        )
        line = (
            alt.Chart(ref_df)
            .mark_line(strokeDash=[6,4])
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart(scatter + line, use_container_width=True)

        # ---------- 2) Absolute error distribution ----------
        hist = (
            alt.Chart(preds_df)
            .mark_bar()
            .encode(
                x=alt.X("abs_error:Q", bin=alt.Bin(maxbins=30), title="Absolute error |r - p|"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count():Q", title="Count")]
            )
            .properties(title="Distribution of absolute errors")
        )
        st.altair_chart(hist, use_container_width=True)

        # ---------- 3) Per-org MAE (top 20 worst) ----------
        org_mae = (
            preds_df.groupby("u_idx", as_index=False)["abs_error"]
            .mean().rename(columns={"abs_error": "mae"})
            .sort_values("mae", ascending=False).head(20)
        )
        chart_org = (
            alt.Chart(org_mae)
            .mark_bar()
            .encode(
                x=alt.X("u_idx:O", sort='-y', title="ORG index"),
                y=alt.Y("mae:Q", title="MAE"),
                tooltip=["u_idx", alt.Tooltip("mae:Q", format=".3f")]
            )
            .properties(title="Organisations with highest MAE (Top 20)")
        )
        st.altair_chart(chart_ex := chart_org, use_container_width=True)

        # ---------- 4) Per-exercise MAE (top 20 worst) ----------
        ex_mae = (
            preds_df.groupby("i_idx", as_index=False)["abs_error"]
            .mean().rename(columns={"abs_error": "mae"})
            .sort_values("mae", ascending=False).head(20)
        )
        chart_ex = (
            alt.Chart(ex_mae)
            .mark_bar()
            .encode(
                x=alt.X("i_idx:O", sort='-y', title="EXID index"),
                y=alt.Y("mae:Q", title="MAE"),
                tooltip=["i_idx", alt.Tooltip("mae:Q", format=".3f")]
            )
            .properties(title="Exercises with highest MAE (Top 20)")
        )
        st.altair_chart(chart_ex, use_container_width=True)

# =============================================================================
# TAB 3 — Recommendations (auto-enriched with org/exercise metadata)
# =============================================================================
with recs_tab:
    st.subheader("Top-N Recommendations per Organisation")

    if "cf_model" not in st.session_state:
        st.info("Train a model first in the Train tab.")
    else:
        model: FunkSVD = st.session_state["cf_model"]
        topn = st.slider("Top-N", 1, 20, 5)
        exclude_seen = st.toggle("Exclude exercises already attempted", value=True)

        # Base predictions
        seen = seen_by_org(train_df)
        u_uniques = np.array(sorted(train_df["ORGID"].unique()))
        i_uniques = np.array(sorted(train_df["EXID"].unique()))

        rows = []
        for u_idx, org_id in enumerate(u_uniques):
            if org_id not in seen:
                continue
            scores = np.asarray(model.predict_for_org(u_idx), dtype=float)
            if exclude_seen:
                for ex in seen[org_id]:
                    j = i2ix.get(ex)
                    if j is not None:
                        scores[j] = -np.inf
            k = min(topn, len(i_uniques))
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            for j in top_idx:
                rows.append({
                    "ORGID": int(org_id),
                    "EXID": int(i_uniques[j]),
                    "Predicted Score": float(scores[j]),
                })

        recs_df = pd.DataFrame(rows)

        # --- Auto-enrich using orgs_full.csv and exercises_full.csv if available ---
        org_meta, ex_meta = load_enrichment(ORG_FILE, EX_FILE)
        if org_meta is not None:
            recs_df = recs_df.merge(org_meta, on="ORGID", how="left")
        if ex_meta is not None:
            recs_df = recs_df.merge(ex_meta, on="EXID", how="left")

        # Optional filter by Industry when present
        if "Industry" in recs_df.columns:
            with st.expander("Filter by Industry", expanded=False):
                industries = sorted([x for x in recs_df["Industry"].dropna().unique()])
                chosen = st.multiselect("Show industries", industries)
                if chosen:
                    recs_df = recs_df[recs_df["Industry"].isin(chosen)].copy()

        # Show the enriched table
        st.dataframe(recs_df, use_container_width=True, height=420)

        st.download_button(
            "Download recommendations (CSV)",
            recs_df.to_csv(index=False).encode("utf-8"),
            file_name="phase2_recommendations_enriched.csv",
            mime="text/csv",
        )

        if not recs_df.empty:
            st.divider()
            st.write("Inspect recommendations by organisation")
            org_sel = st.selectbox("Select ORGID", sorted(recs_df["ORGID"].unique()))
            org_view = (
                recs_df[recs_df["ORGID"] == org_sel]
                .copy()
                .sort_values("Predicted Score", ascending=False)
            )
            st.dataframe(org_view, use_container_width=True)

            # Chart with metadata in tooltips
            tooltips = ["EXID", alt.Tooltip("Predicted Score:Q", format=".3f")]
            if "Industry" in org_view.columns: tooltips.append(alt.Tooltip("Industry:N"))
            if "Org_TTPs" in org_view.columns: tooltips.append(alt.Tooltip("Org_TTPs:N", title="Org TTPs"))
            if "ExTTPs"   in org_view.columns: tooltips.append(alt.Tooltip("ExTTPs:N",   title="Exercise TTPs"))

            chart = (
                alt.Chart(org_view)
                .mark_bar()
                .encode(
                    x=alt.X("EXID:O", sort='-y', title="Exercise ID"),
                    y=alt.Y("Predicted Score:Q", title="Predicted Score (1–5)"),
                    tooltip=tooltips,
                )
                .properties(title=f"Top-{topn} predicted exercises for ORG {org_sel}")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No recommendations were generated. Check that each organisation has some training interactions.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Summary: this page trains a FunkSVD model on provided 1–5 ratings (ExerciseRating), "
    "auto-enriches recommendations with Industry/TTPs, reports RMSE/MAE, "
    "and generates Top-N exercise recommendations per organisation."
)
