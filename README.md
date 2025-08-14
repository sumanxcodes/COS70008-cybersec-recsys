# COS70008 Cybersecurity Exercise Recommendation System

Hey there! This is our group project for COS70008. We’re building a 3-phase recommender system to help suggest the most relevant cybersecurity drills for organisations, based on MITRE ATT&CK tags, past exercise performance, and some cool hybrid ML techniques.

## Project Structure

```
COS70008-cybersec-recsys/
├─ README.md               → You’re reading this!
├─ environment.yml         → Conda env setup with all dependencies
├─ .gitignore               → Files & folders Git won’t track
├─ LICENSE                 → License for the project
├─ CONTRIBUTING.md         → Guidelines for contributing
├─ data/
│  ├─ raw/                  → Original datasets (not tracked in Git)
│  ├─ processed/            → Cleaned & ready-to-use data (also not tracked)
│  └─ .gitkeep               → Keeps empty folders in Git
├─ notebooks/               → Jupyter notebooks for EDA & prototypes
├─ src/                     → All Python source code
│  ├─ data/                 → Data loading & preprocessing
│  ├─ features/             → Feature engineering (e.g., DTM)
│  ├─ models/               → ML models (item-based, hybrid)
│  ├─ eval/                  → Evaluation metrics & scripts
│  └─ dashboard/            → Web dashboard code
├─ tests/                   → Unit/integration tests
├─ scripts/                 → Standalone scripts for data prep, training, etc.
├─ outputs/                 → Generated models, logs, charts (ignored in Git)
└─ .gitkeep                  → Placeholder for empty dirs
```

**Why ignore **``** & **``**?** To avoid huge repos, keep sensitive data private, and ensure everyone generates outputs locally.

## How to Run It (for teammates)

1. **Clone the repo**

```bash
git clone https://github.com/<org-or-user>/COS70008-cybersec-recsys.git
cd COS70008-cybersec-recsys
```

2. **Set up the environment**

```bash
conda env create -f environment.yml
conda activate cybersec-recsys
```

3. **Run Jupyter Notebooks** (for EDA/prototyping)

```bash
python -m ipykernel install --user --name cybersec-recsys --display-name "Python (cybersec-recsys)"
jupyter lab
```

Select the `Python (cybersec-recsys)` kernel.

4. **Run scripts** Example: build a DTM from metadata

```bash
python scripts/build_dtm.py --input data/raw/exercises.csv --out data/processed/dtm.npz
```

5. **Launch the dashboard**

```bash
uvicorn src.dashboard.app:app --reload
# OR
streamlit run src/dashboard/app.py
```

6. **Testing**

```bash
pytest -q
```

---

### Quick Project Goal Recap

- **Phase 1:** Item-based recommender using DTM + similarity scores.
- **Phase 2:** Hybrid recommender (collab filtering + content-based).
- **Phase 3:** Interactive dashboard with APT mapping, history trends, and rec explanations.

Let’s keep things clean, commit often, and document as we go 🚀

