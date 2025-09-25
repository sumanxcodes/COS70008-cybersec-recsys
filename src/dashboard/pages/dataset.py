import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Dataset")
st.caption("Quick previews and column dictionaries, loaded directly from `src/data/`.")

# -----------------------------
# Paths
# -----------------------------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2] if HERE.parent.name == "pages" else HERE.parent  # src/
DATA_DIR = SRC_DIR / "data"

FILES = {
    "orgs":      DATA_DIR / "orgs_full.csv",
    "exercises": DATA_DIR / "exercises_full.csv",
}

# -----------------------------
# Cache + Safe Loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    return pd.read_csv(path)

def safe_read(key):
    path = FILES[key]
    if not path.exists():
        st.error(f"Missing file: `{path}`")
        return None, path
    try:
        df = load_csv(path)
        return df, path
    except Exception as e:
        st.error(f"Failed to load `{path}`\n\n{e}")
        return None, path

# -----------------------------
# Load datasets
# -----------------------------
orgs, orgs_path = safe_read("orgs")
exs,  exs_path  = safe_read("exercises")

# -----------------------------
# Column Dictionaries
# -----------------------------
EXS_DICT = {
    "EXID": "Unique exercise ID",
    "ExCreation": "Creation/version date",
    "ExThreat": "Threat type (e.g., Ransomware, Phishing)",
    "ExTTPs": "MITRE ATT&CK techniques (semicolon-separated)",
    "ExCategories": "Higher-level tags",
    "ExGroups": "Associated adversary groups (e.g., APT29, FIN7)",
    "ExSoftware": "Tools/malware (e.g., Cobalt Strike, Mimikatz)",
    "ExStructure": "Scenario structure (single-phase, multi-stage)",
    "ExMaturity": "1–5 scenario realism/advancement",
    "ExComplexity": "1–5 difficulty/skill requirement",
    "ExLength": "Duration (minutes)",
    "ExAudience": "Intended participants (SOC, management, hybrid)",
    "ExTradeCraftIntra": "Depth within a single technique (variants/evasion)",
    "ExTradeCraftInter": "Cross-technique chaining across stages",
}

ORGS_DICT = {
    "ORGID": "Unique org ID",
    "Industry": "Sector",
    "Region": "Geographic region",
    "Size": "Organisation size classification",
    "SecurityBudget": "Security investment level",
    "PrimarySecurityTeam": "Main operational model",
    "Maturity": "1–5, org cybersecurity posture",
    "Complexity": "1–5, IT/security environment complexity",
    "ExerciseFrequency": "Frequency of running exercises",
    "Threats": "Threat types (semicolon-separated)",
    "TTPs": "Attacker TTPs (semicolon-separated)",
    "Aims": "Objectives/goals of attacks/exercises",
}

# --- unified schema block ---
def schema_block(df: pd.DataFrame, col_dict: dict, *, title: str = "Schema & details"):
    with st.expander(title, expanded=False):
        left, right = st.columns([1.8, 1.2], vertical_alignment="top")

        # Left: column dictionary (only show keys that exist in the file)
        with left:
            st.markdown("**Column dictionary**")
            present = [c for c in col_dict.keys() if c in df.columns]
            dict_df = pd.DataFrame(
                {"Column": present, "Description": [col_dict[c] for c in present]}
            )
            st.dataframe(dict_df, use_container_width=True, hide_index=True)

        # Right: types and missing%
        with right:
            st.markdown("**Types and missing%**")
            meta = pd.DataFrame({
                "Column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "missing%": (df.isna().mean() * 100).round(1).values,
            })
            st.dataframe(meta, use_container_width=True, hide_index=True)

# (optional) expander styling as a card
st.markdown("""
<style>
div.stExpander > details { border: 1px solid rgba(150,150,150,.25); border-radius: 12px; }
div.stExpander > details > summary { background: rgba(255,255,255,.04); padding: 10px 14px; border-radius: 12px; }
div.stExpander p { margin: 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab_exs, tab_orgs, tab_keys = st.tabs(["📘 Exercises", "🏢 Organisations", "🧠 Key interpretations"])

with tab_exs:
    st.subheader("`exercises_full.csv` — Exercise Metadata")
    if exs is not None:
        st.caption(f"📂 `{exs_path}` · shape {exs.shape}")
        st.dataframe(exs, use_container_width=True)
        schema_block(exs, EXS_DICT, title="Schema & details")
    else:
        st.info("Exercises file not found.")

with tab_orgs:
    st.subheader("`orgs_full.csv` — Organisation Profiles")
    if orgs is not None:
        st.caption(f"📂 `{orgs_path}` · shape {orgs.shape}")
        st.dataframe(orgs, use_container_width=True)
        schema_block(orgs, ORGS_DICT, title="Schema & details")
    else:
        st.info("Organisations file not found.")

with tab_keys:
    st.subheader("Key interpretations")
    st.markdown("""
**Org Maturity** vs **Exercise Maturity**
- *Org Maturity*: posture/process capability (context).
- *Exercise Maturity*: scenario realism/impact (content).

**Tradecraft**
- **Intra**: depth within one technique (variants, tools, evasion).
- **Inter**: chaining multiple techniques across the kill chain.

**Practical reads**
- Use **ExThreat + ExTTPs** to drive content similarity.
- Use **Org Threats/TTPs** to align demand vs coverage.
- Keep **ExComplexity/ExMaturity** in range for the org’s capability.
""")
