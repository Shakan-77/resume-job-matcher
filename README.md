# 📄 Resume ↔ Job Description Matcher

Upload a resume (PDF, DOCX or text) and paste a job description. The app tells you:

- an **overall match score** (0-100),
- **section scores** for skills, experience and education,
- **which resume line supports each job requirement** (and which requirements have no evidence),
- **matched, related and missing skills**, e.g. "the JD asks for *Deep learning*; you list *PyTorch*",
- short **improvement suggestions**.

A second tab **ranks several resumes** against one job description.

Everything runs locally with small open-source transformer models
(`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, ~90 MB / ~130 MB).

![ROC-AUC by method](results/auc_comparison.png)

---

## What are sentence embeddings? (plain words)

A sentence-embedding model reads a piece of text and outputs a list of numbers
(384 of them for the models used here) called an **embedding**. The model was trained
so that texts with *similar meaning* get *similar lists of numbers*, even when they use
different words:

| Text A | Text B | Similar? |
|---|---|---|
| "Built CNNs in PyTorch" | "Experience with deep learning frameworks" | yes, although no word is shared |
| "Built CNNs in PyTorch" | "Managed a retail store" | no |

To compare two texts we measure the angle between their embeddings (**cosine similarity**):
1.0 means "same meaning", around 0 means "unrelated". Keyword search (like TF-IDF) only
counts shared words, so it misses the first example above. That is the main reason to use
embeddings for resumes, where the same skill is described in many ways.

## How the matcher works

```
resume ──► split into bullets/sentences ──► embed each line ─┐
                                                              ├─► for every JD requirement, find the
job description ──► split into requirement lines ──► embed ──┘   most similar resume line (= evidence)

both ──► skill vocabulary (≈200 skills + aliases) ──► matched / related / missing skills
                 │
                 ├─ "implies" rules: PyTorch ⇒ Deep learning, PostgreSQL ⇒ SQL
                 ├─ alternatives: "PyTorch or TensorFlow" – having one is enough
                 └─ fuzzy matching with embeddings for everything else

score = 0.5 × requirement coverage + 0.3 × skill coverage + 0.2 × whole-document similarity
```

- **Requirement coverage**: average of the best-match similarity of every JD line, mapped to 0-100.
  Each model has its own calibration because their cosine ranges differ
  (`python evaluate.py --calibrate` prints the percentiles used).
- **Skill coverage**: exact match = 1 point, related = ½ point, missing = 0, divided by the number of JD skills.
- **Sections**: the resume is split into skills / experience / education using headings
  (or, if there are none, by guessing each line), and each section is compared with the matching part of the JD.

## Results

RESULTS_PLACEHOLDER

## Project layout

```
matcher/
  __init__.py   keeps all downloads/temp files inside the project folder
  parsing.py    read PDF/DOCX/TXT, split into chunks and sections
  skills.py     curated skill vocabulary, aliases, "implies" rules
  embedder.py   loads the embedding model (GPU if free, CPU otherwise), caches vectors
  scoring.py    requirement matching, skill matching, scores, suggestions, ranking
  data.py       loads the Hugging Face evaluation dataset
app.py          Gradio web UI (match + rank tabs)
evaluate.py     compares all methods, writes results/
finetune.py     fine-tunes MiniLM on the labelled pairs (optional)
examples/       fictional resumes and job descriptions to try
tests/          pytest tests
```

## How to run

```bash
# 1. create a virtual environment (Python 3.12) and install
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build; plain "pip install torch" for CPU
pip install -r requirements.txt

# 2. start the web app, then open http://127.0.0.1:7860
python app.py

# 3. optional: reproduce the evaluation / fine-tuning
python finetune.py          # ~5-10 min on a laptop GPU, saves models/minilm-resume-jd
python evaluate.py          # writes results/metrics.md and the charts
pytest                      # run the tests
```

The first run downloads the models into `.hf_cache/` inside the project.
Set `MATCHER_DEVICE=cpu` to force CPU if your GPU is busy.

## Limitations (honest notes)

- The labelled dataset is noisy and most pairs are from the *same field* (e.g. an accountant
  resume vs an accounting job labelled "No Fit"), so "does this person fit?" is much harder
  than "is this the same field?". Similarity alone only moderately predicts the labels.
- The skill vocabulary is hand-made and tech/business-focused; skills outside it are only
  covered through line-level semantic matching.
- PDF parsing depends on the PDF having a text layer (scanned images are not OCR'd).
- A score is a screening aid, not a hiring decision.

## Resume bullet ideas

RESUME_PLACEHOLDER
