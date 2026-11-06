# Phase 0 — Audit of existing work and career-focused roadmap

> Purpose: before building anything new, record what this repository already
> contains, how a skeptical hiring manager would read it, and the phased plan
> that turns it into portfolio evidence for Data Analyst, Data Scientist,
> ML Engineer and AI Engineer roles.

---

## 1. What already exists (inventory)

| Area | What is there | Assessment |
|---|---|---|
| Data | `insurance.csv` (1,338 rows × 7 cols, 3 missing BMI, 1 exact duplicate), SHA-256 fingerprinted | Fine, but this is one of the most-used Kaggle datasets |
| Core pipeline | `medcost/data.py`, `medcost/training.py`: preprocessing inside a `ColumnTransformer`, 6 candidates, 5-fold CV on the fitting split only, fit / calibration / test split stratified by smoker | **Correct and leakage-safe.** Strongest asset in the repo |
| Uncertainty | Split-conformal 90% intervals (`conformal_radius`), with subgroup coverage reported | Above typical portfolio level; good interview material |
| Follow-up experiments | `log_experiment.py`, `stability_experiment.py`, `calibrate_log.py`, `evaluate_log.py` + 8 report files | Rigorous, but reads like a lab notebook instead of a finished product |
| App | `app.py` Streamlit, 3 tabs, example profile | Works, but **refits the model on every cold start** and hard-codes metrics |
| Quality | `tests/` (pipeline + Streamlit `AppTest`), GitHub Actions CI, devcontainer | Good foundation |
| Docs | README, `MODEL_CARD.md`, `LEARNING_GUIDE.md` | Thorough, but too defensive (see below) |
| Legacy | `NoteBook.ipynb`, `Report.docx` at repo root. These have known errors: leakage from scaling before the split, wrong object used for Ridge R² | Sitting at the root, these are the first files a visitor opens |

## 2. How a skeptical recruiter or hiring manager reads it today

**Positives that should stay:**
- Methodology: fold-contained preprocessing, a calibration split kept separate from the test set, a baseline, and a written admission that the test set was reused for exploratory work. Most junior portfolios get these wrong.
- Conformal prediction intervals with subgroup coverage. This is a real talking point.
- Tests + CI exist.

**Issues that cost interviews, in priority order:**

1. **Not deployed.** The README says "No public deployment is claimed yet." A recruiter spends about 30 seconds on a repo, and a live link usually matters more than everything else here.
2. **The README leads with disclaimers instead of results.** The caveats are correct, but they are repeated so often that they read as a lack of confidence. Keep one *Limitations* section and let the rest of the README sell the project.
3. **The title doesn't match the repo.** The title says "Smoking Behavior", but smoking classification was dropped. Either rename the project or explicitly reframe smoking as an input feature.
4. **Clutter.** Four experiment scripts and about 15 report files sit at the same level as the product. Consolidate them, and move the legacy notebook and report to `archive/`.
5. **Engineering gaps (this is your stated growth area):**
   - Training runs inside the web app (`st.cache_resource` → `build_bundle()`). Production systems train once and serve a versioned artifact.
   - Artifacts are pickles with no metadata or versioning, and there is no model registry or experiment tracking.
   - There is no prediction API (FastAPI), no input schema validation (pydantic), no container (Docker) and no logging or monitoring.
   - Numbers are hard-coded in `app.py` (877, 1,819, 4,144, 71.2%, 25,101) and duplicated from the reports, so they will drift out of date.
   - Tests assert exact dollar outputs (`'$25,101'`), so they break whenever the scikit-learn version changes.
   - Code style: dense one-liners such as `f=load_data();a,b,c=...`. Reviewers notice this.
6. **Dataset familiarity.** Hiring managers have seen "insurance.csv regression" many times. **Differentiation must come from engineering and communication, not from the model.**
7. **No coverage for Data Analyst or AI Engineer roles.** The repo has no SQL, no BI dashboard, no business framing and no LLM component.

## 3. Strategy

Use this repository as the **ML Engineering / Data Science flagship**. The modelling is already done, so the work that remains is turning it into a deployed, reproducible, tested service. Then add two layers that cover the other roles:

- **AI Engineer layer:** an LLM assistant that *calls the prediction API as a tool* and answers questions grounded in the model card, with an evaluation harness.
- **Data Analyst layer:** a SQL (DuckDB) analysis layer plus a Power BI dashboard. **Decision needed:** this dataset is thin for a DA showcase (7 columns, no time dimension). The recommendation is a small DA layer here plus a separate, richer DA project later.

## 4. Phased roadmap

Every phase delivers code plus an explanation file (`docs/phases/PHASE_N_*.md`). Code comments use these markers:
- `🎯 INTERVIEW FOCUS` — logic you must be able to explain or rebuild from scratch
- `📖 READ ONLY` — boilerplate you should recognise, not memorise

| Phase | Goal | Key skills demonstrated | Roles |
|---|---|---|---|
| **1. Restructure** | Archive legacy files, consolidate experiments into one config-driven training entry point, clean code style, rewrite the README to lead with results | Project structure, config management, technical writing | All |
| **2. Train once, serve artifacts** | Training script → versioned artifact (`skops`/`joblib` + JSON metadata: data hash, metrics, sklearn version). Add MLflow tracking. The app loads the artifact instead of training. Tests check tolerances, not exact strings | Reproducibility, experiment tracking, artifact management | MLE, DS |
| **3. Prediction API** | FastAPI `/predict` and `/health`, pydantic input schema with range validation, API tests | API design, validation, testing | MLE, AI Eng |
| **4. Containerise and deploy** | Dockerfile, CI that builds and tests the image, live deployment (Streamlit Cloud + Render or Hugging Face Spaces), live link in the README | Docker, CI/CD, deployment | MLE |
| **5. Monitoring lite** | Prediction logging, a data-drift report (PSI per feature) against the training distribution | MLOps fundamentals | MLE |
| **6. LLM assistant** | Agent uses the `/predict` tool and retrieves from the model card to explain estimates and limitations. Include an eval set (groundedness, refusal of out-of-scope requests) | Tool calling, RAG, LLM evaluation | AI Eng |
| **7. Analyst layer** | DuckDB SQL analysis (cost drivers by segment, smoker premium by age band), exported tables → Power BI dashboard, plus a one-page business memo | SQL, BI, stakeholder communication | DA |

Order rationale: phases 1–4 fix the biggest recruiter-facing gaps (clutter, no deployment) and target your stated growth area. Phases 6 and 7 depend on the API from phase 3 and the clean data layer from phase 1.

## 5. Open inputs needed from you

1. **Your CV.** It is not in this repository or session. Paste the text into the chat rather than committing it, because the repo is public. It will be used to align the README and the interview stories with your actual background.
2. **Data Analyst decision:** a Power BI layer in this repo only, or a separate DA project as well?
3. **Deployment target preference.** Default if you have none: Streamlit Community Cloud for the UI and Render or Hugging Face Spaces for the API, all free tier.
