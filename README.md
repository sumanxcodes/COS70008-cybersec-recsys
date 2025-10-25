# COS70008 Cybersecurity Exercise Recommendation System

Hey there! 👋
This repository contains our group project for **COS70008 – Technology Innovation Research & Project (2025-HS2)**.
We’re developing a **3-phase Cybersecurity Exercise Recommendation System**, designed to suggest relevant cybersecurity drills for organisations based on **MITRE ATT&CK** data, historical performance, and Deep-learning models.

---

## Project Overview

Cyber exercises are vital for improving an organisation’s cyber resilience — but choosing the *right next exercise* can be tricky.
Our system bridges that gap using a mix of **content-based**, **collaborative**, and **deep learning** techniques:

- **Phase 1:** Baseline recommender using exercise-to-exercise similarity (TF-IDF / cosine).
- **Phase 2:** Collaborative filtering (FunkSVD) to learn org–exercise relationships.
- **Phase 3:** Deep-learning Softmax model for MITRE ATT&CK TTP prediction.
All three are wrapped inside an interactive **Streamlit dashboard**.

---

## 📂 Project Structure

```
COS70008-cybersec-recsys/
├─ README.md                    → You’re reading this file!
├─ environment.yml              → Conda environment for all dependencies
├─ .gitignore                   → Files & folders excluded from Git
├─ LICENSE
├─ CONTRIBUTING.md
│
├─ data/                        → All data files
│  ├─ raw/                     → Original datasets (not tracked in Git)
│  ├─ processed/               → Cleaned & merged data (local only)
│  └─ .gitkeep
│
├─ notebooks/                   → Jupyter notebooks for EDA and prototypes
│
├─ src/                         → Main source code
│  ├─ data/                    → Data loading, parsing, cleaning utilities
│       ├─ ex_sim.npz                   → Exercise-to-exercise similarity matrix (Phase 1)
│       ├─ org_ttp_map.csv              → Mapping of organisations ↔ MITRE TTPs
│       ├─ exercises_full.csv           → Exercise metadata (TTPs, Threats, etc.)
│       ├─ orgs_full.csv                → Organisation profiles and observed TTPs
│       ├─ ratings_train_full.csv       → Training ratings matrix (CF)
│       ├─ ratings_validation_full.csv  → Validation ratings matrix
│       ├─ ratings_test_full.csv        → Test ratings matrix
│       └─ enterprise-attack/           → MITRE ATT&CK STIX knowledge base (JSON)
│  └─ dashboard/                → Streamlit dashboard (Phase 1–3 pages)
│       ├─ app.py               → Main Streamlit entry point
│       ├─ funk_svd.py          → Custom FunkSVD recommender implementation
│       └─ pages/               → Streamlit pages for each project phase
│           ├─ project_brief.py
│           ├─ dataset.py
│           ├─ eda.py
│           ├─ phase_one.py
│           ├─ phase_two.py
│           └─ phase_three.py
│
├─ artifacts/                → Saved model weights, vocab files, and logs
├─ models/                   → Exported TensorFlow and CF models
└─ .gitkeep
```

## Environment Setup

We use **TensorFlow** as our main deep-learning framework.
Create the environment via Conda:

```bash
git clone https://github.com/sumanxcodes/COS70008-cybersec-recsys.git
cd COS70008-cybersec-recsys
conda env create -f environment.yml
conda activate cybersec-recsys
```

---

## 🚀 Running the Project

### 1. Launch Jupyter (for data exploration)

Before running Jupyter Lab, let's copy the MITRE STIX JSON files from the dashboard section into the notebook section. 
```bash
mkdir -p notebooks/attack-stix-data/enterprise-attack/
cp src/data/enterprise-attack/* notebooks/attack-stix-data/enterprise-attack/
```

```bash
python -m ipykernel install --user --name cybersec-recsys --display-name "Python (cybersec-recsys)"
jupyter lab
```
All the notebooks are in the folder COS70008-cybersec-recsys/notebooks

### 2. Start the Streamlit dashboard
```bash
streamlit run src/dashboard/app.py
```
You’ll see tabs for each phase along with EDA and Dataset:

- **Phase 1 – Baseline (DTM + Cosine)**
- **Phase 2 – Collaborative Filtering (FunkSVD)**
- **Phase 3 – Deep Learning (Softmax TTP Prediction)**

---

## 🧩 Phase Summary

### 🏁 Phase 1 – Baseline Recommender
- Built **Document-Term Matrix (DTM)** using exercise metadata (MITRE tags, difficulty, scope).
- Used **TF-IDF** and **cosine similarity** to rank top-N similar exercises.
- Visualised results in an interactive Altair scatter plot.

### 🤝 Phase 2 – Collaborative Filtering (FunkSVD)
- Constructed **Org × Exercise** matrix from rating datasets (`ratings_train_full.csv`, etc.).
- Implemented **matrix factorisation (FunkSVD)** via SGD with bias terms and early stopping.
- Predicted new exercise ratings and recommendations per organisation.
- Integrated evaluation metrics (RMSE, Top-N recall).

### 🧠 Phase 3 – Deep Learning Softmax Model
- Mapped **MITRE ATT&CK TTPs** to organisational attack data.
- Built a **single-tower softmax neural network** using TensorFlow.
- Trained on MITRE-only dataset;
- Used for **TTP prediction** and **gap analysis** (suggesting unseen but likely techniques).
- Fully integrated into the Streamlit dashboard for visual insight.

---
## 📊 Datasets Used

| Dataset | Description |
|----------|--------------|
| `exercises_full.csv` | Exercise metadata (name, difficulty, MITRE tags, etc.) |
| `orgs_full.csv` | Organisation profiles with observed TTPs |
| `ratings_train_full.csv` / `ratings_validation_full.csv` / `ratings_test_full.csv` | Org × Exercise rating matrices for CF |
| `enterprise-attack.json` | MITRE ATT&CK STIX knowledge base |
| `org_ttp_map.csv` | Mapping between organisations and TTP techniques |

---

## 🧩 Tech Stack

**Core Libraries:**
- TensorFlow / Keras (Deep learning)
- NumPy, Pandas, PyArrow (data handling)
- Scikit-learn (TF-IDF, evaluation metrics)
- Streamlit + Altair (interactive dashboard & visualisation)

**Framework:**
- Python 3.11 (via Conda `cybersec-recsys` environment)

---
