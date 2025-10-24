# pages/phase_three.py
# =============================================================================
# Phase 3 — Deep Learning (Softmax) for TTP prediction (MITRE-only → model)
# =============================================================================

from __future__ import annotations
import os, io, json, random, collections, zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np, collections
import pandas as pd
import streamlit as st

# Silence TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf
except Exception:
    tf = None  # we'll gate training/inference on this

# -----------------------------------------------------------------------------
# Project paths (match your Phase-2 pattern)
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2] if HERE.parent.name == "pages" else HERE.parent
DATA_DIR = SRC_DIR / "data"
ARTI_DIR = SRC_DIR / "artifacts"
MODELS_DIR = SRC_DIR / "models"
ARTI_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Correct default STIX path (no sidebar upload/toggle)
DEFAULT_STIX = DATA_DIR / "enterprise-attack" / "enterprise-attack.json"

st.set_page_config(page_title="Phase 3 — Deep Learning (Softmax)", layout="wide")
st.title("Phase 3 — Deep Learning (Softmax) for TTPs")
st.caption("End-to-end pipeline: build vocabularies → create training rows → split → train a softmax model → evaluate → recommend missing TTPs.")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ext_id(o: dict):
    for ref in (o.get("external_references") or []):
        if ref.get("source_name") in ("mitre-attack", "mitre-ics-attack", "mitre-mobile-attack"):
            if "external_id" in ref:
                return ref["external_id"]
    return None

def _parent_of(tid: str):
    return tid.split(".")[0] if tid and "." in tid else None

def _ensure_list(x):
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    if pd.isna(x): return []
    try: return list(x)
    except Exception: return []

def _multi_hot(indices, size: int):
    v = np.zeros(size, dtype=np.float32)
    for ix in set(indices or []):
        j = int(ix)
        if 0 <= j < size: v[j] = 1.0
    return v

def idx_to_tid(ix, ix2tech: dict) -> str:
    """Safe key access whether ix2tech keys are ints or strings."""
    if ix in ix2tech: return ix2tech[ix]
    i = int(ix)
    if i in ix2tech: return ix2tech[i]
    s = str(i)
    if s in ix2tech: return ix2tech[s]
    raise KeyError(f"Technique index {ix} not in ix2tech")

def _first_match(cols, candidates):
    colset = set(map(str, cols))
    for c in candidates:
        if c in colset:
            return c
    return None

# -----------------------------------------------------------------------------
# Cached loaders / builders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_stix(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    objs = [o for o in bundle.get("objects", []) if isinstance(o, dict)]
    by_id = {o.get("id"): o for o in objs if o.get("id")}
    return objs, by_id

@st.cache_data(show_spinner=False)
def build_vocabularies(stix_path: str | Path):
    objs, _ = load_stix(stix_path)
    techniques, tactics, groups, software = [], [], [], []
    platforms = set()

    for o in objs:
        t = o.get("type")
        if t == "attack-pattern":
            tid = _ext_id(o)
            if tid:
                techniques.append({"id": tid})
                for p in (o.get("x_mitre_platforms") or []):
                    platforms.add(p)
        elif t == "x-mitre-tactic":
            taid = _ext_id(o)
            if taid: tactics.append({"id": taid})
        elif t == "intrusion-set":
            gid = _ext_id(o)
            if gid: groups.append({"id": gid})
        elif t in ("malware", "tool"):
            sid = _ext_id(o)
            if sid: software.append({"id": sid})

    def make_vocab(items):
        ids = sorted({it["id"] for it in items})
        return {k: i for i, k in enumerate(ids)}, {i: k for i, k in enumerate(ids)}

    tech2ix, ix2tech = make_vocab(techniques)
    tac2ix, _ = make_vocab(tactics)
    grp2ix, _ = make_vocab(groups)
    sft2ix, _ = make_vocab(software)
    plat2ix = {p: i for i, p in enumerate(sorted(platforms))}

    return dict(
        tech2ix=tech2ix, ix2tech=ix2tech,
        tac2ix=tac2ix, grp2ix=grp2ix, sft2ix=sft2ix, plat2ix=plat2ix,
        counts=dict(
            techniques=len(tech2ix), tactics=len(tac2ix),
            groups=len(grp2ix), software=len(sft2ix), platforms=len(plat2ix)
        )
    )

@st.cache_data(show_spinner=False)
def build_kb_train(stix_path: str | Path, vocabs: dict) -> pd.DataFrame:
    objs, by_id = load_stix(stix_path)
    tech2ix, tac2ix, grp2ix, sft2ix, plat2ix = (
        vocabs["tech2ix"], vocabs["tac2ix"], vocabs["grp2ix"], vocabs["sft2ix"], vocabs["plat2ix"]
    )

    # tactic shortname -> tactic id
    short2id = {}
    for o in objs:
        if o.get("type") == "x-mitre-tactic":
            taid = _ext_id(o)
            short = o.get("x_mitre_shortname")
            if taid and short: short2id[short] = taid

    tech_platforms, tech_tactics = {}, {}
    for o in objs:
        if o.get("type") == "attack-pattern":
            tid = _ext_id(o)
            if not tid: continue
            tech_platforms[tid] = set(o.get("x_mitre_platforms") or [])
            tacs = set()
            for kp in (o.get("kill_chain_phases") or []):
                if kp.get("kill_chain_name") in ("mitre-attack","mitre-enterprise-attack","mitre-mobile-attack"):
                    taid = short2id.get(kp.get("phase_name"))
                    if taid: tacs.add(taid)
            tech_tactics[tid] = tacs

    group_to_techs, software_to_techs = collections.defaultdict(set), collections.defaultdict(set)
    for o in objs:
        if o.get("type") != "relationship" or o.get("relationship_type") != "uses": continue
        src = by_id.get(o.get("source_ref")); dst = by_id.get(o.get("target_ref"))
        if not src or not dst: continue
        if src.get("type") in ("intrusion-set","malware","tool") and dst.get("type") == "attack-pattern":
            owner, tech = _ext_id(src), _ext_id(dst)
            if owner and tech:
                if src.get("type") == "intrusion-set": group_to_techs[owner].add(tech)
                else: software_to_techs[owner].add(tech)

    pop_count = collections.Counter()
    for Ts in group_to_techs.values(): pop_count.update(Ts)
    for Ts in software_to_techs.values(): pop_count.update(Ts)

    def enc_list(elems, vocab): return sorted({vocab[e] for e in elems if e in vocab})
    def enc_single(e, vocab):   return vocab.get(e, -1)

    rows = []
    def emit(techs, g=None, s=None):
        for t in techs:
            ctx = set(techs) - {t}
            if not ctx: continue
            ctx_tacs, ctx_plats = set(), set()
            for c in ctx:
                ctx_tacs |= tech_tactics.get(c, set())
                ctx_plats |= tech_platforms.get(c, set())
            rows.append(dict(
                target_ix=tech2ix.get(t, -1),
                ctx_ttps_ix=enc_list(ctx, tech2ix),
                group_ix=enc_single(g, grp2ix) if g else -1,
                software_ix=enc_single(s, sft2ix) if s else -1,
                tactic_ixs=enc_list(ctx_tacs, tac2ix),
                platform_ixs=enc_list(ctx_plats, plat2ix),
                popularity=float(pop_count.get(t, 0)),
                parent_ix=tech2ix.get(_parent_of(t), -1) if _parent_of(t) else -1,
            ))
    for g, Ts in group_to_techs.items(): emit(Ts, g=g)
    for s, Ts in software_to_techs.items(): emit(Ts, s=s)

    return pd.DataFrame([r for r in rows if r["target_ix"] >= 0])

@st.cache_data(show_spinner=False)
def leave_owner_out_split(df: pd.DataFrame, seed=42, frac_group=0.2, frac_soft=0.2):
    random.seed(seed)
    groups_present = sorted(set(df["group_ix"]) - {-1})
    soft_present   = sorted(set(df["software_ix"]) - {-1})

    def sample(ids, frac, s):
        if not ids: return set()
        k = max(1, int(len(ids)*frac))
        rng = random.Random(s)
        return set(rng.sample(ids, k))

    val_g = sample(groups_present, frac_group, seed+1)
    val_s = sample(soft_present,   frac_soft,  seed+2)

    is_val = df["group_ix"].isin(val_g) | df["software_ix"].isin(val_s)
    df_tr, df_va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    meta = dict(
        seed=seed,
        counts=dict(rows_total=len(df), rows_train=len(df_tr), rows_val=len(df_va),
                    groups_total=len(groups_present), groups_val=len(val_g),
                    software_total=len(soft_present), software_val=len(val_s))
    )
    return df_tr, df_va, meta

# -----------------------------------------------------------------------------
# Model training utils
# -----------------------------------------------------------------------------
def _safe_tf():
    if tf is None:
        st.error("TensorFlow not installed. Install with `pip install tensorflow` (or `tensorflow-macos` on Apple Silicon).")
        return False
    return True

def _df_to_arrays(df: pd.DataFrame, sizes: Dict[str,int], grp2ix: Dict, sft2ix: Dict):
    D_TEC, D_TAC, D_PLT = sizes["D_TEC"], sizes["D_TAC"], sizes["D_PLT"]
    df = df.copy()
    for col in ["ctx_ttps_ix","tactic_ixs","platform_ixs"]:
        df[col] = df[col].apply(_ensure_list)
    g = np.clip(df["group_ix"].fillna(-1).astype("int32").to_numpy(), 0, len(grp2ix))
    s = np.clip(df["software_ix"].fillna(-1).astype("int32").to_numpy(), 0, len(sft2ix))
    ctx  = np.stack([_multi_hot(xs, D_TEC) for xs in df["ctx_ttps_ix"]]).astype("float32")
    tac  = np.stack([_multi_hot(xs, D_TAC) for xs in df["tactic_ixs"]]).astype("float32")
    plat = np.stack([_multi_hot(xs, D_PLT) for xs in df["platform_ixs"]]).astype("float32")
    y    = df["target_ix"].astype("int32").to_numpy()
    return dict(group=g, software=s, ctx=ctx, tac=tac, plat=plat, y=y)

def _make_tfds(arrs, batch=512, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        ({"group_ix": arrs["group"], "software_ix": arrs["software"],
          "ctx": arrs["ctx"], "tac": arrs["tac"], "plat": arrs["plat"]},
         arrs["y"])
    )
    if shuffle:
        ds = ds.shuffle(min(10000, arrs["y"].shape[0]), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def _build_model(D_TEC, D_TAC, D_PLT, D_G, D_S,
                 d_g=32, d_s=32, d_tac=16, d_plt=8, d_ctx=128, hidden=256, l2=1e-5, dropout=0.1):
    inp_g  = tf.keras.Input(shape=(),      dtype=tf.int32,   name="group_ix")
    inp_s  = tf.keras.Input(shape=(),      dtype=tf.int32,   name="software_ix")
    inp_ctx= tf.keras.Input(shape=(D_TEC,),dtype=tf.float32, name="ctx")
    inp_tac= tf.keras.Input(shape=(D_TAC,),dtype=tf.float32, name="tac")
    inp_pl = tf.keras.Input(shape=(D_PLT,),dtype=tf.float32, name="plat")

    emb_g = tf.keras.layers.Embedding(D_G+1, d_g, embeddings_regularizer=tf.keras.regularizers.l2(l2))(inp_g)
    emb_s = tf.keras.layers.Embedding(D_S+1, d_s, embeddings_regularizer=tf.keras.regularizers.l2(l2))(inp_s)

    proj_ctx = tf.keras.layers.Dense(d_ctx, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(inp_ctx)
    proj_tac = tf.keras.layers.Dense(d_tac, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(inp_tac)
    proj_pl  = tf.keras.layers.Dense(d_plt, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(inp_pl)

    x = tf.keras.layers.Concatenate()([emb_g, emb_s, proj_ctx, proj_tac, proj_pl])
    x = tf.keras.layers.Dense(hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    logits = tf.keras.layers.Dense(D_TEC, name="logits")(x)

    model = tf.keras.Model(inputs={"group_ix": inp_g, "software_ix": inp_s, "ctx": inp_ctx, "tac": inp_tac, "plat": inp_pl},
                           outputs=logits)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10"),
        ],
    )
    return model

# -----------------------------------------------------------------------------
# Robust model loading with mtime cache
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model_with_mtime(path_str: str, mtime: float):
    return tf.keras.models.load_model(path_str)

def find_existing_model() -> Optional[Path]:
    p1 = MODELS_DIR / "kb_softmax_tf.keras"
    p2 = MODELS_DIR / "kb_softmax_tf_final.keras"
    return p1 if p1.exists() else (p2 if p2.exists() else None)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
train_tab, eval_tab, recs_tab = st.tabs(["Train", "Evaluate", "Recommendations"])

# =============================================================================
# TAB 1 — Train (Steps 1–5)
# =============================================================================
with train_tab:
    st.header("Training Pipeline (Steps 1–5)")
    st.caption("Build vocabularies from STIX, generate training rows, create a split, and train a softmax classifier.")
    if not DEFAULT_STIX.exists():
        st.error(f"Default STIX not found at: {DEFAULT_STIX}")
        st.stop()

    # Step 1 — Build vocabularies
    st.subheader("Step 1 — Build vocabularies")
    st.caption("Extract numeric IDs for techniques, tactics, groups, software, and platforms.")
    c1, c2 = st.columns(2)
    if c1.button("Build vocabs"):
        voc = build_vocabularies(DEFAULT_STIX)
        for k, v in voc.items():
            if k == "counts": continue
            (ARTI_DIR / f"{k}.json").write_text(json.dumps(v))
        st.session_state["voc"] = voc
        st.success(
            f"Saved to artifacts/. "
            f"{voc['counts']['techniques']} techniques, {voc['counts']['tactics']} tactics, "
            f"{voc['counts']['groups']} groups, {voc['counts']['software']} software, {voc['counts']['platforms']} platforms."
        )

    if c2.button("Download vocabs (.zip)"):
        if "voc" not in st.session_state:
            st.warning("Build vocabs first.")
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for k, v in st.session_state["voc"].items():
                    if k == "counts": continue
                    z.writestr(f"{k}.json", json.dumps(v))
            st.download_button("⬇️ Save vocabs.zip", buf.getvalue(), "vocabs.zip", mime="application/zip")

    # Step 2 — KB rows
    st.subheader("Step 2 — Create KB training rows")
    st.caption("Build supervised examples from 'uses' relations: predict a target TTP from its context.")
    if st.button("Build kb_train.parquet"):
        voc = st.session_state.get("voc") or build_vocabularies(DEFAULT_STIX)
        df = build_kb_train(DEFAULT_STIX, voc)
        out = ARTI_DIR / "kb_train.parquet"
        df.to_parquet(out, index=False)
        st.session_state["kb_df"] = df
        st.success(f"Saved {len(df):,} rows → {out}")
        st.dataframe(df.head(10), use_container_width=True)
        covered = set(df["target_ix"].unique()); total = set(voc["tech2ix"].values())
        st.info(f"Coverage: {len(covered)} / {len(total)} techniques ({len(covered)/len(total)*100:.1f}%).")

    # Step 3 — Split
    st.subheader("Step 3 — Leave-owner-out split")
    st.caption("Hold out a subset of groups/software so validation checks generalisation to unseen owners.")
    cA, cB, cC = st.columns(3)
    seed = cA.number_input("Seed", value=42, step=1)
    frac_g = cB.slider("Val fraction (groups)", 0.05, 0.5, 0.2, 0.05)
    frac_s = cC.slider("Val fraction (software)", 0.05, 0.5, 0.2, 0.05)

    if st.button("Split now"):
        # explicit, no ambiguous 'or'
        df = st.session_state.get("kb_df", None)
        if df is None and (ARTI_DIR / "kb_train.parquet").exists():
            df = pd.read_parquet(ARTI_DIR / "kb_train.parquet")
        if df is None or df.empty:
            st.warning("kb_train.parquet not found/built yet.")
        else:
            tr, va, meta = leave_owner_out_split(df, seed=int(seed), frac_group=float(frac_g), frac_soft=float(frac_s))
            tr.to_parquet(ARTI_DIR / "kb_train_split.parquet", index=False)
            va.to_parquet(ARTI_DIR / "kb_val.parquet", index=False)
            (ARTI_DIR / "kb_split_meta.json").write_text(json.dumps(meta, indent=2))
            st.session_state["kb_tr"] = tr; st.session_state["kb_va"] = va
            st.success(f"Train rows: {len(tr):,} | Val rows: {len(va):,}")
            st.json(meta)

    # Step 4 & 5 — Build & Train
    st.subheader("Step 4 & 5 — Build & train softmax model")
    st.caption("Train a multi-class softmax that predicts a missing technique given context features.")
    if _safe_tf():
        c1, c2, c3 = st.columns(3)
        epochs   = c1.number_input("Epochs", value=20, min_value=1, step=1)
        batch    = c2.number_input("Batch size", value=512, min_value=32, step=32)
        patience = c3.number_input("Early stopping patience", value=3, min_value=1, step=1)

        if st.button("Train model"):
            voc = st.session_state.get("voc") or build_vocabularies(DEFAULT_STIX)
            tr  = st.session_state.get("kb_tr", None)
            va  = st.session_state.get("kb_va", None)
            if tr is None and (ARTI_DIR / "kb_train_split.parquet").exists():
                tr = pd.read_parquet(ARTI_DIR / "kb_train_split.parquet")
            if va is None and (ARTI_DIR / "kb_val.parquet").exists():
                va = pd.read_parquet(ARTI_DIR / "kb_val.parquet")

            if tr is None or va is None or tr.empty or va.empty:
                st.warning("Run Step 3 split first.")
            else:
                sizes = dict(D_TEC=len(voc["tech2ix"]), D_TAC=len(voc["tac2ix"]), D_PLT=len(voc["plat2ix"]),
                             D_G=len(voc["grp2ix"]), D_S=len(voc["sft2ix"]))
                arr_tr = _df_to_arrays(tr, sizes, voc["grp2ix"], voc["sft2ix"])
                arr_va = _df_to_arrays(va, sizes, voc["grp2ix"], voc["sft2ix"])
                ds_tr  = _make_tfds(arr_tr, batch=int(batch), shuffle=True)
                ds_va  = _make_tfds(arr_va, batch=int(batch), shuffle=False)

                model = _build_model(**sizes)
                st.text(model.summary())

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_top10", mode="max", patience=int(patience), restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(filepath=str(MODELS_DIR / "kb_softmax_tf.keras"),
                                                       monitor="val_top10", mode="max", save_best_only=True),
                    tf.keras.callbacks.CSVLogger(str(ARTI_DIR / "kb_train_log.csv")),
                ]
                # Build weight per target technique index
                counts = collections.Counter(tr["target_ix"].astype(int).tolist())
                # Inverse-sqrt weighting to avoid huge weights
                class_weight = {int(k): float(1.0 / np.sqrt(v + 1)) for k, v in counts.items()}

                st.write(f"Using {len(class_weight):,} class weights (inverse sqrt of frequency). Example:", list(class_weight.items())[:5])

                # --- Train model with class weights ---
                _ = model.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=int(epochs),
                    callbacks=callbacks,
                    verbose=1,
                    class_weight=class_weight,     # ✅ add this line
                )
                model.save(MODELS_DIR / "kb_softmax_tf_final.keras")
                st.success("Training complete. Best checkpoint + final model saved in /models")

                metrics = model.evaluate(ds_va, return_dict=True, verbose=0)
                a,b,c = st.columns(3)
                a.metric("Val Top-1", f"{metrics['top1']:.3f}")
                b.metric("Val Top-5", f"{metrics['top5']:.3f}")
                c.metric("Val Top-10", f"{metrics['top10']:.3f}")

# =============================================================================
# TAB 2 — Evaluate (Quantitative results for trained model)
# =============================================================================
with eval_tab:
    st.header("Evaluate Trained Model")
    st.caption("Review saved checkpoints, training curves, and quantitative metrics on the validation split.")

    # --- Show model directory and available files ---
    st.code(f"MODELS_DIR = {MODELS_DIR.resolve()}")
    try:
        st.write("Files in models:", sorted(os.listdir(MODELS_DIR)))
    except Exception:
        st.warning("Models directory not accessible.")

    # --- Detect trained model file ---
    model_path = find_existing_model()
    if model_path is None:
        st.info("No model file found. Train in the Train tab first.")
        st.stop()

    # --- Load model with mtime-based cache ---
    mtime = model_path.stat().st_mtime
    model = load_trained_model_with_mtime(str(model_path), mtime)
    st.success(f"Loaded model from: {model_path.name}")
    st.text(model.summary())

    # --- Display training log if available ---
    st.subheader("📈 Training history")
    st.caption("CSV logs from training (epochs, loss, and top-K accuracy).")
    log_path = ARTI_DIR / "kb_train_log.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        st.dataframe(df_log.tail(10), use_container_width=True)

        import altair as alt
        c1 = (
            alt.Chart(df_log)
            .mark_line(point=True)
            .encode(
                x="epoch:Q",
                y="val_top10:Q",
                tooltip=["epoch", "val_top1", "val_top5", "val_top10", "val_loss"]
            )
            .properties(title="Validation Top-10 Accuracy over epochs")
        )
        c2 = (
            alt.Chart(df_log)
            .mark_line(point=True, color="orange")
            .encode(x="epoch:Q", y="val_loss:Q")
            .properties(title="Validation Loss")
        )
        st.altair_chart(c1, use_container_width=True)
        st.altair_chart(c2, use_container_width=True)
    else:
        st.info("No training log found (kb_train_log.csv). Train once to generate it.")

    # --- Quantitative evaluation on validation data ---
    st.subheader("📊 Evaluate on Validation Split")
    st.caption("Compute Top-1/5/10 accuracy and loss on the held-out validation set.")
    voc = st.session_state.get("voc") or build_vocabularies(DEFAULT_STIX)
    va_path = ARTI_DIR / "kb_val.parquet"
    if not va_path.exists():
        st.warning("Validation split not found. Run split in Train tab.")
    else:
        df_va = pd.read_parquet(va_path)
        sizes = dict(D_TEC=len(voc["tech2ix"]), D_TAC=len(voc["tac2ix"]), D_PLT=len(voc["plat2ix"]),
                     D_G=len(voc["grp2ix"]), D_S=len(voc["sft2ix"]))
        arr_va = _df_to_arrays(df_va, sizes, voc["grp2ix"], voc["sft2ix"])
        ds_va  = _make_tfds(arr_va, batch=512, shuffle=False)
        metrics = model.evaluate(ds_va, return_dict=True, verbose=0)
        st.json(metrics)

        c1, c2, c3 = st.columns(3)
        c1.metric("Top-1 Accuracy", f"{metrics['top1']:.3f}")
        c2.metric("Top-5 Accuracy", f"{metrics['top5']:.3f}")
        c3.metric("Top-10 Accuracy", f"{metrics['top10']:.3f}")

    # --- Optional: Confusion / top-prediction inspection ---
    with st.expander("🔍 Inspect sample predictions"):
        st.caption("Peek at the model’s top-5 guesses for random validation rows.")
        sample_n = st.slider("Number of samples to preview", 5, 50, 10)
        df_val_sample = pd.read_parquet(va_path).sample(min(sample_n, len(pd.read_parquet(va_path)))) if va_path.exists() else None
        if df_val_sample is not None:
            arr_sample = _df_to_arrays(df_val_sample, sizes, voc["grp2ix"], voc["sft2ix"])
            xb = {
                "group_ix": arr_sample["group"],
                "software_ix": arr_sample["software"],
                "ctx": arr_sample["ctx"],
                "tac": arr_sample["tac"],
                "plat": arr_sample["plat"],
            }
            logits = model(xb, training=False).numpy()
            probs = tf.nn.softmax(logits).numpy()
            preds = np.argsort(-probs, axis=1)[:, :5]
            rows = []
            for i, y_true in enumerate(arr_sample["y"]):
                top = [idx_to_tid(ix, voc["ix2tech"]) for ix in preds[i]]
                true_tid = idx_to_tid(int(y_true), voc["ix2tech"])
                rows.append(dict(true=true_tid, top_predicted="; ".join(top)))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =============================================================================
# TAB 3 — Recommendations (Step 6)
# =============================================================================
with recs_tab:
    st.header("Recommendations — Org Gap Analysis (Step 6)")
    st.caption("Given known techniques for an org, suggest plausible missing TTPs with optional heuristic bonuses.")

    # Auto-load model
    force_reload_rec = st.button("🔄 Force reload model (Recommendations)")
    model_path = find_existing_model()
    if model_path is None:
        st.info("Train a model first in the Train tab.")
        st.stop()
    if force_reload_rec:
        try: st.cache_resource.clear()
        except Exception: pass
    model = load_trained_model_with_mtime(str(model_path), model_path.stat().st_mtime)
    st.success(f"Using model: {model_path.name}")

    # Build metadata from STIX
    if not DEFAULT_STIX.exists():
        st.error(f"Default STIX not found at: {DEFAULT_STIX}")
        st.stop()

    objs, _ = load_stix(DEFAULT_STIX)
    short2id, tacid2name = {}, {}
    for o in objs:
        if o.get("type") == "x-mitre-tactic":
            taid = _ext_id(o)
            if taid:
                short2id[o.get("x_mitre_shortname")] = taid
                tacid2name[taid] = o.get("name","")
    tech2tactics, tech2name = {}, {}
    for o in objs:
        if o.get("type") == "attack-pattern":
            tid = _ext_id(o)
            if not tid: continue
            tech2name[tid] = o.get("name","")
            tacs=set()
            for kp in (o.get("kill_chain_phases") or []):
                if kp.get("kill_chain_name") in ("mitre-attack","mitre-enterprise-attack","mitre-mobile-attack"):
                    taid = short2id.get(kp.get("phase_name"))
                    if taid: tacs.add(taid)
            tech2tactics[tid]=tacs

    voc = st.session_state.get("voc") or build_vocabularies(DEFAULT_STIX)
    tech2ix, ix2tech, tac2ix = voc["tech2ix"], voc["ix2tech"], voc["tac2ix"]
    D_TEC, D_TAC, D_PLT = len(tech2ix), len(tac2ix), len(voc["plat2ix"])

    # --- Single org demo ---
    with st.expander("Single org demo", expanded=True):
        st.caption("Enter a semicolon-separated list of known techniques; we’ll propose the next most likely ones.")
        known_str = st.text_input("Known TTPs (semicolon-separated)", value="T1218;T1046")
        topk = st.slider("Top-K", 5, 20, 10)
        new_tactic_bonus = st.slider("Bonus: adds new tactic", 0.0, 0.5, 0.10, 0.01)
        parent_bonus     = st.slider("Bonus: parent technique", 0.0, 0.5, 0.05, 0.01)

        if st.button("Recommend"):
            known_ids = [t.strip() for t in known_str.split(";") if t.strip()]
            seen_ix = [tech2ix[t] for t in known_ids if t in tech2ix]
            if not seen_ix:
                st.warning("No valid known TTPs mapped to vocab.")
            else:
                seen_tacs=set()
                for t in known_ids: seen_tacs |= tech2tactics.get(t, set())
                xb = {
                    "group_ix":   np.array([0], dtype="int32"),
                    "software_ix":np.array([0], dtype="int32"),
                    "ctx":  np.expand_dims(_multi_hot(seen_ix, D_TEC), 0),
                    "tac":  np.expand_dims(_multi_hot([tac2ix[ta] for ta in seen_tacs if ta in tac2ix], D_TAC), 0),
                    "plat": np.expand_dims(np.zeros(D_PLT, dtype=np.float32), 0),
                }
                logits = model(xb, training=False).numpy()[0]
                probs  = tf.nn.softmax(logits).numpy()
                for ix in seen_ix: probs[ix] = -1.0
                final = probs.copy()
                for ix, p in enumerate(probs):
                    if p < 0: continue
                    tid = idx_to_tid(ix, ix2tech)
                    new_t = tech2tactics.get(tid, set()) - seen_tacs
                    if new_t: final[ix] += float(new_tactic_bonus) * len(new_t)
                    if _parent_of(tid) is None: final[ix] += float(parent_bonus)
                top_ix = np.argsort(-final)[:int(topk)]
                rows = []
                for ix in top_ix:
                    tid = idx_to_tid(ix, ix2tech)
                    tacs = [tacid2name.get(ta, ta) for ta in sorted(tech2tactics.get(tid, set()))]
                    why=[]
                    new_t = tech2tactics.get(tid, set()) - seen_tacs
                    if new_t: why.append("adds new tactic(s)")
                    if _parent_of(tid) is None: why.append("parent technique")
                    if not why: why.append("co-occurs in MITRE")
                    rows.append(dict(tech_id=tid, tech_name=tech2name.get(tid,""), tactics=", ".join(tacs), why="; ".join(why)))
                st.write("**Known TTPs**"); st.code(", ".join(known_ids))
                st.write("**Predicted missing TTPs**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Batch over org_ttp_map.csv (show results inline, no CSV download) ---
    with st.expander("Batch over org_ttp_map.csv", expanded=False):
        st.caption("Provide ORGID → known TTPs. We’ll compute Top-K candidates per org and display them inline.")
        # Auto-detect default; allow upload to override
        default_orgmap_path = DATA_DIR / "org_ttp_map.csv"
        org_map_df = None
        if default_orgmap_path.exists():
            try:
                _df = pd.read_csv(default_orgmap_path)
                org_col = _first_match(_df.columns, ["ORGID", "orgid", "OrgID", "org_id"])
                ttp_col = _first_match(_df.columns, ["attack_id", "TTP", "ttp", "tech_id", "TechniqueID"])
                if org_col and ttp_col:
                    org_map_df = _df.rename(columns={org_col: "ORGID", ttp_col: "attack_id"})[["ORGID", "attack_id"]]
                    st.success(f"Using default: {default_orgmap_path.name} ({len(org_map_df):,} rows)")
                else:
                    st.info("Default org_ttp_map.csv found but columns not recognised.")
            except Exception as e:
                st.info(f"Could not read default org_ttp_map.csv: {e}")

        up = st.file_uploader("Upload org_ttp_map.csv (optional)", type=["csv"])
        if up is not None:
            try:
                _df = pd.read_csv(up)
                org_col = _first_match(_df.columns, ["ORGID", "orgid", "OrgID", "org_id"])
                ttp_col = _first_match(_df.columns, ["attack_id", "TTP", "ttp", "tech_id", "TechniqueID"])
                if org_col and ttp_col:
                    org_map_df = _df.rename(columns={org_col: "ORGID", ttp_col: "attack_id"})[["ORGID", "attack_id"]]
                    st.success(f"Loaded uploaded file ({len(org_map_df):,} rows)")
                else:
                    st.error("Please include columns for ORGID and attack_id (or common variants).")
            except Exception as e:
                st.error(f"Upload read error: {e}")

        if org_map_df is None or org_map_df.empty:
            st.info("No org_ttp_map.csv loaded. Put one in /data or upload above.")
        else:
            topk_b = st.slider("Top-K (batch)", 5, 20, 10, key="topk_batch")

            if st.button("Run batch"):
                df_map = org_map_df.copy()
                out_rows = []
                for org_id, sub in df_map.groupby("ORGID"):
                    known = [t for t in sub["attack_id"].dropna().astype(str).tolist() if t.startswith("T")]
                    seen_ix = [tech2ix[t] for t in known if t in tech2ix]
                    if not seen_ix:
                        continue
                    seen_tacs = set()
                    for t in known:
                        seen_tacs |= tech2tactics.get(t, set())
                    xb = {
                        "group_ix":   np.array([0], dtype="int32"),
                        "software_ix":np.array([0], dtype="int32"),
                        "ctx":  np.expand_dims(_multi_hot(seen_ix, D_TEC), 0),
                        "tac":  np.expand_dims(_multi_hot([tac2ix[ta] for ta in seen_tacs if ta in tac2ix], D_TAC), 0),
                        "plat": np.expand_dims(np.zeros(D_PLT, dtype=np.float32), 0),
                    }
                    logits = model(xb, training=False).numpy()[0]
                    probs  = tf.nn.softmax(logits).numpy()
                    for ix in seen_ix:
                        probs[ix] = -1.0
                    final = probs.copy()
                    for ix, p in enumerate(probs):
                        if p < 0:
                            continue
                        tid = idx_to_tid(ix, ix2tech)
                        new_t = tech2tactics.get(tid, set()) - seen_tacs
                        if new_t:
                            final[ix] += 0.10 * len(new_t)
                        if _parent_of(tid) is None:
                            final[ix] += 0.05
                    top_ix = np.argsort(-final)[:int(topk_b)]
                    for rank, ix in enumerate(top_ix, 1):
                        tid = idx_to_tid(ix, ix2tech)
                        tacs = [tacid2name.get(ta, ta) for ta in sorted(tech2tactics.get(tid, set()))]
                        out_rows.append(dict(
                            ORGID=int(org_id),
                            rank=int(rank),
                            tech_id=tid,
                            tech_name=tech2name.get(tid, ""),
                            tactics=", ".join(tacs),
                        ))

                res = pd.DataFrame(out_rows)

                # 🔎 Simple interactive view
                st.subheader("Batch results")
                st.caption("Filter by ORGID and sort columns to explore recommendations quickly.")
                if not res.empty:
                    # Optional: filter by ORG
                    with st.expander("Filter/Sort", expanded=False):
                        orgs = sorted(res["ORGID"].unique().tolist())
                        pick = st.multiselect("Show ORGID(s)", orgs)
                        if pick:
                            res = res[res["ORGID"].isin(pick)].copy()
                        sort_by = st.selectbox("Sort by", ["ORGID", "rank", "tech_id", "tech_name"])
                        res = res.sort_values([sort_by, "ORGID"] if sort_by != "ORGID" else ["ORGID", "rank"])

                    st.dataframe(res, use_container_width=True)
                else:
                    st.info("No recommendations produced (check that ORGIDs have at least one known TTP).")

st.divider()
st.caption("Artifacts in /artifacts; models in /models. Default STIX is loaded from /data/enterprise-attack/enterprise-attack.json.")
