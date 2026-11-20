"""Score how well a resume matches a job description.

The idea, step by step:

1. Cut both documents into short chunks (bullets / sentences).
2. Embed every chunk (turn it into a vector that captures its meaning).
3. For each job requirement, find the resume chunk with the highest cosine
   similarity -> that is the "evidence" for the requirement.
4. Extract skills from both documents with a curated vocabulary. JD skills
   missing from the resume get a second chance through (a) "implies" rules
   (PyTorch -> Deep learning) and (b) embedding similarity to the resume's
   skills and lines (fuzzy matching).
5. Combine requirement coverage, skill coverage and whole-document similarity
   into one 0-100 score, plus per-section scores and suggestions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from matcher.embedder import Embedder, get_embedder
from matcher.parsing import is_education_line, split_chunks, split_sections
from matcher.skills import count_skills, extract_skills, implied_by, lines_mentioning


# --------------------------------------------------------------------------- #
# Calibration: raw cosine similarity -> 0..100
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSettings:
    """Per-model numbers. Different models produce different cosine ranges
    (bge-small rarely goes below 0.4, MiniLM often goes near 0), so a raw
    cosine is mapped linearly from [low, high] onto [0, 100].
    The values were chosen by looking at the similarity distribution on the
    training split of the evaluation dataset (``python evaluate.py --calibrate``):
    ``low`` sits near the 5th percentile of requirement-to-best-line cosines,
    ``high`` a bit above the 95th, so a typical strong line scores ~70+.
    """

    low: float  # line-level cosine that maps to a score of 0
    high: float  # line-level cosine that maps to a score of 100
    doc_low: float  # whole-document cosines run higher, so they get their own range
    doc_high: float
    skill_threshold: float  # cosine above which two skill names count as related
    line_threshold: float  # cosine above which a resume line counts as evidence for a skill


MODEL_SETTINGS = {
    "minilm": ModelSettings(low=0.15, high=0.65, doc_low=0.35, doc_high=0.85, skill_threshold=0.60, line_threshold=0.55),
    "bge-small": ModelSettings(low=0.50, high=0.80, doc_low=0.75, doc_high=0.95, skill_threshold=0.80, line_threshold=0.75),
}
MODEL_SETTINGS["finetuned"] = MODEL_SETTINGS["minilm"]

# How much each component contributes to the overall score.
WEIGHTS = {"requirements": 0.5, "skills": 0.3, "document": 0.2}

STRONG, PARTIAL = 70.0, 40.0  # requirement score bands (0-100)

# JD lines that are about the company/benefits rather than the job itself.
_BOILERPLATE = re.compile(
    r"equal opportunity|e-verify|benefits|401\(?k\)?|paid time off|pto\b|salary|compensation|health insurance|"
    r"dental|vision insurance|apply (now|today)|click apply|visa sponsorship|disability|veteran|"
    r"race, color|religion|sexual orientation|award-winning|recognized by|founded in",
    re.IGNORECASE,
)


def settings_for(embedder: Embedder) -> ModelSettings:
    return MODEL_SETTINGS.get(embedder.name, MODEL_SETTINGS["minilm"])


def to_score(sim: np.ndarray | float, low: float, high: float) -> np.ndarray | float:
    """Map cosine similarity linearly from [low, high] to 0..100 (clipped)."""
    return np.clip((np.asarray(sim) - low) / (high - low), 0.0, 1.0) * 100.0


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #
@dataclass
class RequirementMatch:
    requirement: str  # a line from the job description
    evidence: str  # best-matching resume line
    similarity: float  # raw cosine
    score: float  # 0..100

    @property
    def status(self) -> str:
        return "strong" if self.score >= STRONG else "partial" if self.score >= PARTIAL else "weak"


@dataclass
class SkillMatch:
    skill: str  # skill named in the JD
    status: str  # "matched" | "related" | "missing"
    evidence: str = ""  # why we think so (resume skill / line)
    similarity: float | None = None


@dataclass
class MatchResult:
    overall: float
    components: dict[str, float | None]
    sections: dict[str, float | None]
    requirements: list[RequirementMatch]
    skills: list[SkillMatch]
    resume_skills: list[str]
    suggestions: list[str] = field(default_factory=list)

    def skills_with(self, status: str) -> list[SkillMatch]:
        return [s for s in self.skills if s.status == status]

    @property
    def matched_skills(self) -> list[str]:
        return [s.skill for s in self.skills_with("matched")]

    @property
    def related_skills(self) -> list[str]:
        return [s.skill for s in self.skills_with("related")]

    @property
    def missing_skills(self) -> list[str]:
        return [s.skill for s in self.skills_with("missing")]


# --------------------------------------------------------------------------- #
# Document analysis (cached: the same resume is often scored against many JDs)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResumeInfo:
    chunks: tuple[str, ...]
    sections: dict
    skills: tuple[str, ...]


@dataclass(frozen=True)
class JobInfo:
    requirements: tuple[str, ...]
    education_lines: tuple[str, ...]
    skill_lines: tuple[str, ...]
    skills: tuple[str, ...]


@lru_cache(maxsize=4096)
def analyze_resume(text: str) -> ResumeInfo:
    chunks = tuple(split_chunks(text))
    sections = split_sections(text, skill_counter=count_skills)
    return ResumeInfo(chunks=chunks, sections=sections, skills=tuple(extract_skills(text)))


@lru_cache(maxsize=4096)
def analyze_job(text: str) -> JobInfo:
    reqs = [c for c in split_chunks(text, min_words=4) if not _BOILERPLATE.search(c)]
    if not reqs:  # very short JD: use whatever is there
        reqs = split_chunks(text, min_words=1) or [text.strip()]
    return JobInfo(
        requirements=tuple(reqs),
        education_lines=tuple(r for r in reqs if is_education_line(r)),
        skill_lines=tuple(r for r in reqs if count_skills(r) > 0),
        skills=tuple(extract_skills(text)),
    )


def resume_doc_text(text: str) -> str:
    """Resume as one cleaned string (chunks joined) for whole-document embedding."""
    return " ".join(analyze_resume(text).chunks) or text


def job_doc_text(text: str) -> str:
    """Job description without company/benefits boilerplate, as one string."""
    return " ".join(analyze_job(text).requirements) or text


# --------------------------------------------------------------------------- #
# Core matching
# --------------------------------------------------------------------------- #
def best_matches(queries: list[str], candidates: list[str], embedder: Embedder) -> tuple[np.ndarray, np.ndarray]:
    """For every query, the index and cosine of the most similar candidate."""
    if not queries or not candidates:
        return np.zeros(len(queries), dtype=int), np.zeros(len(queries))
    sims = embedder.encode(queries) @ embedder.encode(candidates).T  # (n_queries, n_candidates)
    return sims.argmax(axis=1), sims.max(axis=1)


def coverage(queries: list[str], candidates: list[str], embedder: Embedder) -> float | None:
    """Average 0..100 score of how well ``candidates`` cover ``queries`` (None if nothing to cover)."""
    if not queries:
        return None
    if not candidates:
        return 0.0
    _, sims = best_matches(queries, candidates, embedder)
    s = settings_for(embedder)
    return float(np.mean(to_score(sims, s.low, s.high)))


def document_similarity(resume_chunks: list[str], job_chunks: list[str], embedder: Embedder) -> float:
    """Cosine between the average chunk embeddings of the two documents."""
    if not resume_chunks or not job_chunks:
        return 0.0
    r = embedder.encode(resume_chunks).mean(axis=0)
    j = embedder.encode(job_chunks).mean(axis=0)
    return float(r @ j / (np.linalg.norm(r) * np.linalg.norm(j) + 1e-9))


def _alternatives(job: JobInfo) -> dict[str, list[str]]:
    """Skills offered as alternatives in one JD line, e.g. "PyTorch or TensorFlow".

    Having any one of them satisfies the others, so they shouldn't all be reported missing.
    """
    alts: dict[str, list[str]] = {}
    for line in job.requirements:
        if re.search(r"\bor\b", line, re.IGNORECASE):
            group = extract_skills(line)
            for skill in group:
                alts.setdefault(skill, []).extend(o for o in group if o != skill)
    return alts


def match_skills(job: JobInfo, resume: ResumeInfo, embedder: Embedder) -> list[SkillMatch]:
    """Classify every JD skill as matched / related / missing."""
    s = settings_for(embedder)
    have = set(resume.skills)
    implied = implied_by(list(resume.skills))
    results: list[SkillMatch] = []
    unresolved: list[str] = []

    alternatives = _alternatives(job)
    for skill in job.skills:
        alt = next((o for o in alternatives.get(skill, []) if o in have), None)
        if skill in have:
            line = next(iter(lines_mentioning(skill, list(resume.chunks))), "")
            results.append(SkillMatch(skill, "matched", evidence=line or skill, similarity=1.0))
        elif alt:
            results.append(SkillMatch(skill, "related", evidence=f"you have {alt}, listed as an alternative"))
        elif skill in implied:
            results.append(SkillMatch(skill, "related", evidence=f"{implied[skill]} (implies {skill})"))
        else:
            unresolved.append(skill)

    # Fuzzy matching with embeddings for the rest.
    if unresolved:
        resume_skill_names = list(resume.skills)
        skill_idx, skill_sim = best_matches(unresolved, resume_skill_names, embedder)
        line_idx, line_sim = best_matches(unresolved, list(resume.chunks), embedder)
        for i, skill in enumerate(unresolved):
            if resume_skill_names and skill_sim[i] >= s.skill_threshold:
                ev, sim = resume_skill_names[skill_idx[i]], skill_sim[i]
                results.append(SkillMatch(skill, "related", evidence=f"similar to your skill '{ev}'", similarity=float(sim)))
            elif resume.chunks and line_sim[i] >= s.line_threshold:
                results.append(SkillMatch(skill, "related", evidence=resume.chunks[line_idx[i]], similarity=float(line_sim[i])))
            else:
                results.append(SkillMatch(skill, "missing"))

    order = {k: i for i, k in enumerate(job.skills)}
    return sorted(results, key=lambda m: order[m.skill])


def match(resume_text: str, job_text: str, embedder: Embedder | None = None) -> MatchResult:
    """Compare one resume with one job description."""
    embedder = embedder or get_embedder()
    s = settings_for(embedder)
    resume, job = analyze_resume(resume_text), analyze_job(job_text)
    chunks = list(resume.chunks) or [resume_text.strip() or "(empty)"]

    # 1) Requirement -> best resume evidence.
    reqs = list(job.requirements)
    idx, sims = best_matches(reqs, chunks, embedder)
    scores = to_score(sims, s.low, s.high)
    requirements = [
        RequirementMatch(req, chunks[i], float(sim), float(sc)) for req, i, sim, sc in zip(reqs, idx, sims, scores)
    ]

    # 2) Skills.
    skills = match_skills(job, resume, embedder)
    if skills:
        credit = sum(1.0 if m.status == "matched" else 0.5 if m.status == "related" else 0.0 for m in skills)
        skill_cov: float | None = 100.0 * credit / len(skills)
    else:
        skill_cov = None

    # 3) Whole-document similarity.
    doc = float(to_score(document_similarity(chunks, reqs, embedder), s.doc_low, s.doc_high))

    components = {"requirements": float(np.mean(scores)) if len(scores) else 0.0, "skills": skill_cov, "document": doc}
    overall = combine(components)

    # 4) Sections.
    sec = resume.sections
    sections = {
        # Each JD skill name vs the resume's skill names and skills-section lines,
        # so "TensorFlow" still gets partial credit from "PyTorch".
        "skills": coverage(list(job.skills), list(resume.skills) + list(sec["skills"]), embedder) if job.skills else None,
        "experience": coverage([r for r in reqs if r not in job.education_lines], list(sec["experience"]) or chunks, embedder),
        "education": coverage(list(job.education_lines), list(sec["education"]), embedder),
    }

    result = MatchResult(
        overall=overall,
        components=components,
        sections=sections,
        requirements=requirements,
        skills=skills,
        resume_skills=list(resume.skills),
    )
    result.suggestions = make_suggestions(result, resume)
    return result


def combine(components: dict[str, float | None]) -> float:
    """Weighted average of the available components (a missing one is skipped)."""
    present = {k: v for k, v in components.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in present)
    return round(sum(WEIGHTS[k] * v for k, v in present.items()) / total_w, 1) if total_w else 0.0


# --------------------------------------------------------------------------- #
# Suggestions
# --------------------------------------------------------------------------- #
def make_suggestions(result: MatchResult, resume: ResumeInfo, max_items: int = 8) -> list[str]:
    tips: list[str] = []

    missing = result.missing_skills
    if missing:
        tips.append(
            "Missing keywords from the job description: "
            + ", ".join(missing[:8])
            + ". If you have used them, add a bullet that shows where and how."
        )
    for m in result.skills_with("related")[:3]:
        tips.append(f"The JD asks for **{m.skill}**; you show related experience ({m.evidence[:80]}). Name it explicitly.")

    weak = sorted((r for r in result.requirements if r.status == "weak"), key=lambda r: r.score)
    for r in weak[:3]:
        tips.append(f"No clear evidence for the requirement: “{r.requirement[:120]}”. Add a matching achievement if true.")

    if result.sections.get("education") == 0.0:
        tips.append("The job mentions education/certification requirements, but no education section was found.")

    numeric = sum(bool(re.search(r"\d", c)) for c in resume.chunks)
    if resume.chunks and numeric / len(resume.chunks) < 0.25:
        tips.append("Few bullets contain numbers. Quantify impact (e.g. 'cut latency 30%', 'served 10k users').")

    if result.overall >= 75:
        tips.append("Strong match overall: mirror the job title and top 3 requirements in your summary line.")
    elif result.overall < 40:
        tips.append("Low overlap: this role may need skills you don't show yet, or the resume needs heavy tailoring.")
    return tips[:max_items]


# --------------------------------------------------------------------------- #
# Rank mode
# --------------------------------------------------------------------------- #
def rank_resumes(job_text: str, resumes: dict[str, str], embedder: Embedder | None = None) -> list[tuple[str, MatchResult]]:
    """Score several resumes against one JD, best first."""
    embedder = embedder or get_embedder()
    scored = [(name, match(text, job_text, embedder)) for name, text in resumes.items()]
    return sorted(scored, key=lambda item: item[1].overall, reverse=True)
