"""Measure how well each scoring method separates good from poor resume/JD fits.

Methods compared on the held-out test split:
  * TF-IDF cosine            - classic keyword baseline (no neural network)
  * Skill overlap            - share of the JD's vocabulary skills found in the resume
  * <model> whole-doc        - one embedding per document, cosine similarity
  * <model> matcher          - this project's chunk + skill matcher (0-100 score)
  * <model> matcher (learned)- same components, weights fitted with logistic regression on train

Metrics (higher is better):
  * ROC-AUC Good vs No  - probability a random "Good Fit" pair scores above a random "No Fit" pair
  * ROC-AUC Fit vs No   - same, with Good + Potential counted as positives
  * Spearman            - rank correlation between the score and the 3-level label

Usage:
  python evaluate.py                       # minilm + bge-small (+ finetuned if trained)
  python evaluate.py --limit 300           # quick run on a subset
  python evaluate.py --calibrate           # print similarity percentiles used to set ModelSettings
"""

from __future__ import annotations

import argparse
import json

from matcher import RESULTS_DIR  # isort: skip  (sets HF_HOME first)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from matcher.data import LABELS, load_pairs
from matcher.embedder import FINETUNED_DIR, get_embedder
from matcher.scoring import (
    analyze_job,
    analyze_resume,
    best_matches,
    document_similarity,
    job_doc_text,
    match,
    resume_doc_text,
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    good_vs_no = y != 1
    return {
        "auc_good_vs_no": roc_auc_score(y[good_vs_no] == 2, scores[good_vs_no]),
        "auc_fit_vs_no": roc_auc_score(y > 0, scores),
        "spearman": spearmanr(scores, y).statistic,
    }


# --------------------------------------------------------------------------- #
# Scoring methods
# --------------------------------------------------------------------------- #
def tfidf_scores(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=50_000)
    vec.fit(pd.concat([train["resume"], train["job"]]).drop_duplicates())
    r, j = vec.transform(test["resume"]), vec.transform(test["job"])  # rows are L2-normalised
    return np.asarray(r.multiply(j).sum(axis=1)).ravel()


def skill_overlap_scores(df: pd.DataFrame) -> np.ndarray:
    out = []
    for resume, job in zip(df["resume"], df["job"]):
        job_skills = set(analyze_job(job).skills)
        resume_skills = set(analyze_resume(resume).skills)
        out.append(len(job_skills & resume_skills) / len(job_skills) if job_skills else 0.0)
    return np.array(out)


def whole_doc_scores(df: pd.DataFrame, model: str) -> np.ndarray:
    emb = get_embedder(model)
    # Cleaned text (bullets joined, JD boilerplate removed); models truncate long input (256-512 tokens).
    r = emb.encode([resume_doc_text(t) for t in df["resume"]])
    j = emb.encode([job_doc_text(t) for t in df["job"]])
    return (r * j).sum(axis=1)


def matcher_components(df: pd.DataFrame, model: str, desc: str) -> pd.DataFrame:
    """Run the full matcher on every pair and keep the overall score + its components."""
    emb = get_embedder(model)
    rows = []
    for resume, job in tqdm(list(zip(df["resume"], df["job"])), desc=desc, mininterval=10):
        res = match(resume, job, emb)
        rows.append({"overall": res.overall, **{k: (np.nan if v is None else v) for k, v in res.components.items()}})
    return pd.DataFrame(rows)


def learned_combiner(train_c: pd.DataFrame, y_train: np.ndarray, test_c: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Fit component weights with logistic regression (Fit vs No Fit) on train; score test."""
    cols = ["requirements", "skills", "document"]
    X_tr, X_te = train_c[cols].fillna(0).to_numpy(), test_c[cols].fillna(0).to_numpy()
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train > 0)
    return clf.predict_proba(X_te)[:, 1], dict(zip(cols, clf.coef_[0].round(4).tolist()))


# --------------------------------------------------------------------------- #
# Calibration helper
# --------------------------------------------------------------------------- #
def calibrate(train: pd.DataFrame, models: list[str], n: int = 600) -> None:
    """Print percentiles of requirement-level and document-level cosines per model."""
    sample = train.sample(n=min(n, len(train)), random_state=0)
    for model in models:
        emb = get_embedder(model)
        req_sims, doc_sims = [], []
        for resume, job in zip(sample["resume"], sample["job"]):
            r, j = analyze_resume(resume), analyze_job(job)
            _, sims = best_matches(list(j.requirements), list(r.chunks), emb)
            req_sims.extend(sims.tolist())
            doc_sims.append(document_similarity(resume, job, emb))
        q = [5, 10, 25, 50, 75, 90, 95]
        print(f"{model}: requirement best-match cosine percentiles {dict(zip(q, np.percentile(req_sims, q).round(3)))}")
        print(f"{model}: document cosine percentiles           {dict(zip(q, np.percentile(doc_sims, q).round(3)))}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_report(results: dict, test: pd.DataFrame, best_scores: tuple[str, np.ndarray], extra: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    counts = test["label"].value_counts()
    lines = [
        "# Evaluation: does the score separate good fits from poor ones?",
        "",
        f"Dataset: [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit), "
        f"test split ({len(test)} pairs: " + ", ".join(f"{counts.get(lbl, 0)} {lbl}" for lbl in LABELS) + "). "
        "Test job descriptions never appear in the training split.",
        "",
        "| Method | ROC-AUC Good vs No | ROC-AUC Fit vs No | Spearman |",
        "|---|---:|---:|---:|",
    ]
    best_auc = max(m["auc_good_vs_no"] for m in results.values())
    for name, m in results.items():
        bold = "**" if m["auc_good_vs_no"] == best_auc else ""
        lines.append(
            f"| {name} | {bold}{m['auc_good_vs_no']:.3f}{bold} | {m['auc_fit_vs_no']:.3f} | {m['spearman']:.3f} |"
        )
    lines += [
        "",
        "*ROC-AUC*: chance that a random better-fitting pair gets a higher score than a random worse one "
        "(0.5 = coin flip, 1.0 = perfect). *Spearman*: rank correlation with the 3-level label.",
        "",
    ]
    if extra.get("learned_weights"):
        lines += ["## Learned component weights (logistic regression on train, per 1 point of 0-100 score)", ""]
        for model, w in extra["learned_weights"].items():
            lines.append(f"- {model}: " + ", ".join(f"{k} {v:+.4f}" for k, v in w.items()))
        lines.append("")
    lines += ["![ROC-AUC by method](auc_comparison.png)", "", "![Score by label](score_by_label.png)", ""]
    (RESULTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps({"results": results, **extra}, indent=2), encoding="utf-8")

    _plot_auc(results)
    _plot_by_label(best_scores, test)


def _plot_auc(results: dict) -> None:
    names = list(results)
    aucs = [results[n]["auc_good_vs_no"] for n in names]
    order = np.argsort(aucs)
    best = int(np.argmax(aucs))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.2))
    # Baselines in gray, embedding methods in blue, so the comparison reads at a glance.
    colors = ["#2a78d6" if ("matcher" in names[i] or "whole-doc" in names[i]) else "#a9a8a2" for i in order]
    ax.barh([names[i] for i in order], [aucs[i] for i in order], color=colors, height=0.6)
    for y, i in enumerate(order):
        ax.text(aucs[i] + 0.005, y, f"{aucs[i]:.3f}", va="center", fontsize=9, color="#0b0b0b",
                fontweight="bold" if i == best else "normal")
    ax.axvline(0.5, color="#52514e", lw=1, ls="--")
    ax.text(0.503, len(names) - 0.55, "coin flip", fontsize=8, color="#52514e")
    ax.set_xlim(0.45, min(1.0, max(aucs) + 0.08))
    ax.set_xlabel("ROC-AUC, Good Fit vs No Fit (test set)")
    ax.set_title("How well each method separates good from poor fits", loc="left", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#e5e4df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "auc_comparison.png", dpi=150)
    plt.close(fig)


def _plot_by_label(best_scores: tuple[str, np.ndarray], test: pd.DataFrame) -> None:
    name, scores = best_scores
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = [scores[test["y"].to_numpy() == k] for k in range(3)]
    parts = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                       medianprops={"color": "#0b0b0b", "lw": 2})
    for patch in parts["boxes"]:
        patch.set_facecolor("#9fc3ef")
        patch.set_edgecolor("#2a78d6")
    ax.set_yticks([1, 2, 3], LABELS)
    ax.set_xlabel(f"Score: {name}")
    ax.set_title("Score distribution by true label (test set)", loc="left", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#e5e4df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "score_by_label.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="embedding models (default: minilm bge-small [finetuned])")
    parser.add_argument("--limit", type=int, default=None, help="evaluate on a random subset of the test split")
    parser.add_argument("--no-learned", action="store_true", help="skip the logistic-regression combiner (faster)")
    parser.add_argument("--calibrate", action="store_true", help="print similarity percentiles and exit")
    parser.add_argument("--merge", action="store_true",
                        help="keep results already in results/metrics.json and add/overwrite these models")
    parser.add_argument("--skip-baselines", action="store_true", help="don't recompute TF-IDF / skill overlap")
    args = parser.parse_args()

    models = args.models or ["minilm", "bge-small"] + (["finetuned"] if FINETUNED_DIR.exists() else [])
    train, test = load_pairs("train"), load_pairs("test")
    if args.calibrate:
        calibrate(train, models)
        return
    if args.limit:
        test = test.sample(n=args.limit, random_state=0).reset_index(drop=True)
    y = test["y"].to_numpy()

    results: dict[str, dict] = {}
    extra: dict = {"learned_weights": {}}
    if args.merge and (RESULTS_DIR / "metrics.json").exists():
        old = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        results, extra["learned_weights"] = old["results"], old.get("learned_weights", {})
    scores: dict[str, np.ndarray] = {}  # raw scores from this run (used for the by-label chart)

    def add(name: str, values: np.ndarray) -> None:
        scores[name], results[name] = values, metrics(values, y)

    if not args.skip_baselines:
        add("TF-IDF cosine (baseline)", tfidf_scores(train, test))
        add("Skill overlap (vocabulary only)", skill_overlap_scores(test))
        print({k: round(v["auc_good_vs_no"], 3) for k, v in results.items()})

    for model in models:
        add(f"{model} whole-doc", whole_doc_scores(test, model))
        comps = matcher_components(test, model, desc=f"{model} matcher (test)")
        add(f"{model} matcher", comps["overall"].to_numpy())
        # The fine-tuned model has seen the train pairs, so fitting weights on them would be biased.
        if not args.no_learned and model != "finetuned":
            train_comps = matcher_components(train, model, desc=f"{model} matcher (train)")
            learned, weights = learned_combiner(train_comps, train["y"].to_numpy(), comps)
            add(f"{model} matcher (learned weights)", learned)
            extra["learned_weights"][model] = weights
        print({k: round(v["auc_good_vs_no"], 3) for k, v in results.items()})

    best = max(scores, key=lambda n: results[n]["auc_good_vs_no"])
    write_report(results, test, (best, scores[best]), extra)
    print((RESULTS_DIR / "metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
