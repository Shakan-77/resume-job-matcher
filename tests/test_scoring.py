"""End-to-end scoring tests with the small MiniLM model (downloaded once into .hf_cache)."""

import pytest

from matcher import EXAMPLES_DIR
from matcher.scoring import combine, match, rank_resumes, to_score


def read(name):
    return (EXAMPLES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def embedder():
    from matcher.embedder import Embedder

    return Embedder("minilm", device="cpu")  # CPU keeps tests independent of the shared GPU


def test_to_score_clips_to_0_100():
    assert to_score(0.0, 0.2, 0.8) == 0
    assert to_score(1.0, 0.2, 0.8) == 100
    assert to_score(0.5, 0.2, 0.8) == pytest.approx(50)


def test_combine_skips_missing_components():
    weights = {"requirements": 0.5, "skills": 0.3, "document": 0.2}
    assert combine({"requirements": 60.0, "skills": None, "document": 60.0}, weights) == 60.0
    assert combine({"requirements": 100.0, "skills": 0.0, "document": 0.0}, weights) == 50.0


def test_match_result_structure(embedder):
    res = match(read("resume_ml_engineer.txt"), read("job_ml_engineer.txt"), embedder)
    assert 0 <= res.overall <= 100
    assert set(res.sections) == {"skills", "experience", "education"}
    assert res.requirements and all(r.evidence for r in res.requirements)
    assert "Python" in res.matched_skills and "PyTorch" in res.matched_skills
    # PyTorch implies deep learning, so it must not be reported as missing.
    assert "Deep learning" not in res.missing_skills
    assert "Kubernetes" in res.missing_skills
    assert res.suggestions


def test_alternatives_are_not_missing(embedder):
    # "PyTorch or TensorFlow": having PyTorch is enough.
    res = match(read("resume_ml_engineer.txt"), read("job_ml_engineer.txt"), embedder)
    assert "TensorFlow" in res.related_skills


def test_evidence_points_to_relevant_line(embedder):
    res = match(read("resume_ml_engineer.txt"), read("job_ml_engineer.txt"), embedder)
    bert = next(r for r in res.requirements if "transformer models" in r.requirement)
    assert "BERT" in bert.evidence


def test_matching_resume_ranks_first(embedder):
    resumes = {n: read(n) for n in ("resume_ml_engineer.txt", "resume_frontend_dev.txt", "resume_data_analyst.txt")}
    ml_rank = rank_resumes(read("job_ml_engineer.txt"), resumes, embedder)
    assert ml_rank[0][0] == "resume_ml_engineer.txt"
    da_rank = rank_resumes(read("job_data_analyst.txt"), resumes, embedder)
    assert da_rank[0][0] == "resume_data_analyst.txt"
    assert ml_rank[0][1].overall > ml_rank[-1][1].overall + 20


def test_empty_resume_does_not_crash(embedder):
    res = match("", read("job_data_analyst.txt"), embedder)
    assert res.overall < 30
