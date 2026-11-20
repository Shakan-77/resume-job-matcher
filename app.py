"""Gradio web app: match a resume to a job description, or rank several resumes.

Run with:  python app.py   then open http://127.0.0.1:7860
"""

from __future__ import annotations

from pathlib import Path

from matcher import EXAMPLES_DIR  # isort: skip  (sets HF_HOME first)

import gradio as gr
import pandas as pd

from matcher.embedder import DEFAULT_MODEL, FINETUNED_DIR, get_embedder
from matcher.parsing import read_document
from matcher.scoring import MatchResult, match, rank_resumes

MODEL_CHOICES = ["minilm", "bge-small"] + (["finetuned"] if FINETUNED_DIR.exists() else [])
STATUS_ICON = {"strong": "🟢", "partial": "🟡", "weak": "🔴"}


def _example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text(encoding="utf-8")


def _resume_text(file_path: str | None, pasted: str) -> str:
    """Prefer an uploaded file; fall back to pasted text."""
    if file_path:
        text = read_document(file_path)
        if not text.strip():
            raise gr.Error("Couldn't read any text from that file (is it a scanned image PDF?). Paste the text instead.")
        return text
    return pasted or ""


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}/100"


def _verdict(score: float) -> str:
    if score >= 70:
        return "Strong match"
    if score >= 50:
        return "Good partial match"
    if score >= 35:
        return "Weak match"
    return "Poor match"


def _render(result: MatchResult):
    """Turn a MatchResult into the values shown by the UI components."""
    c = result.components
    banner = (
        f"## {result.overall:.0f}/100 · {_verdict(result.overall)}\n"
        f"Requirement coverage **{_fmt(c['requirements'])}** · "
        f"Skill coverage **{_fmt(c['skills'])}** · "
        f"Overall similarity **{_fmt(c['document'])}**"
    )
    sections = pd.DataFrame(
        [{"Section": k.title(), "Score (0-100)": None if v is None else round(v, 1)} for k, v in result.sections.items()]
    )

    def chips(items: list[str]) -> str:
        return ", ".join(f"`{s}`" for s in items) if items else "_none_"

    related = result.skills_with("related")
    skills_md = (
        f"**✅ Matched ({len(result.matched_skills)}):** {chips(result.matched_skills)}\n\n"
        f"**🔶 Related ({len(related)}):** "
        + ("; ".join(f"`{m.skill}` ← {m.evidence[:70]}" for m in related) if related else "_none_")
        + f"\n\n**❌ Missing ({len(result.missing_skills)}):** {chips(result.missing_skills)}"
    )
    reqs = pd.DataFrame(
        [
            {
                "": STATUS_ICON[r.status],
                "Job requirement": r.requirement,
                "Best evidence in resume": r.evidence,
                "Score": round(r.score),
            }
            for r in result.requirements
        ]
    )
    tips = "\n".join(f"- {t}" for t in result.suggestions) or "- Looks good!"
    return banner, sections, skills_md, reqs, tips


def run_match(file_path: str | None, pasted: str, job: str, model: str):
    resume = _resume_text(file_path, pasted)
    if not resume.strip():
        raise gr.Error("Upload a resume file or paste resume text.")
    if not job.strip():
        raise gr.Error("Paste a job description.")
    return _render(match(resume, job, get_embedder(model)))


def run_rank(job: str, files: list[str] | None, model: str):
    if not job.strip():
        raise gr.Error("Paste a job description.")
    if not files:
        raise gr.Error("Upload two or more resumes (PDF, DOCX or TXT).")
    resumes = {Path(f).name: read_document(f) for f in files}
    ranked = rank_resumes(job, resumes, get_embedder(model))
    rows = []
    for pos, (name, r) in enumerate(ranked, start=1):
        rows.append(
            {
                "Rank": pos,
                "Resume": name,
                "Score": r.overall,
                "Requirements": round(r.components["requirements"], 1),
                "Skills": None if r.components["skills"] is None else round(r.components["skills"], 1),
                "Matched skills": ", ".join(r.matched_skills[:8]),
                "Missing skills": ", ".join(r.missing_skills[:8]),
            }
        )
    return pd.DataFrame(rows)


with gr.Blocks(title="Resume ↔ Job Matcher") as demo:
    gr.Markdown(
        "# 📄 Resume ↔ Job Description Matcher\n"
        "Uses **sentence embeddings** (small transformer models) to compare a resume with a job description "
        "line by line: each job requirement is matched to the resume line that best supports it, skills are "
        "checked against a curated vocabulary with fuzzy matching, and everything is combined into a 0-100 score."
    )
    model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Embedding model")

    with gr.Tab("Match one resume"):
        with gr.Row():
            with gr.Column():
                resume_file = gr.File(label="Resume (PDF / DOCX / TXT)", file_types=[".pdf", ".docx", ".txt"], type="filepath")
                resume_text = gr.Textbox(label="…or paste resume text", lines=12)
            job_text = gr.Textbox(label="Job description", lines=19)
        btn = gr.Button("Match", variant="primary")
        banner = gr.Markdown()
        with gr.Row():
            sections = gr.Dataframe(label="Section scores", scale=1)
            skills = gr.Markdown(label="Skills")
        tips = gr.Markdown(label="Suggestions")
        reqs = gr.Dataframe(label="Requirement → best resume evidence", wrap=True)
        gr.Examples(
            [
                [_example("resume_ml_engineer.txt"), _example("job_ml_engineer.txt")],
                [_example("resume_data_analyst.txt"), _example("job_data_analyst.txt")],
                [_example("resume_frontend_dev.txt"), _example("job_ml_engineer.txt")],
            ],
            inputs=[resume_text, job_text],
        )
        btn.click(run_match, [resume_file, resume_text, job_text, model], [banner, sections, skills, reqs, tips])

    with gr.Tab("Rank candidates"):
        gr.Markdown("Paste one job description and upload several resumes; they are ranked by match score.")
        rank_job = gr.Textbox(label="Job description", lines=10, value=_example("job_ml_engineer.txt"))
        rank_files = gr.File(
            label="Resumes", file_count="multiple", file_types=[".pdf", ".docx", ".txt"], type="filepath",
            value=[str(p) for p in sorted(EXAMPLES_DIR.glob("resume_*.txt"))],
        )
        rank_btn = gr.Button("Rank", variant="primary")
        ranking = gr.Dataframe(label="Ranking", wrap=True)
        rank_btn.click(run_rank, [rank_job, rank_files, model], [ranking])


if __name__ == "__main__":
    demo.launch()
