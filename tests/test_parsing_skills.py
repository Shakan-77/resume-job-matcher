"""Fast tests that need no model download: parsing and skill extraction."""

import docx
import pytest

from matcher.parsing import read_document, split_chunks, split_sections
from matcher.skills import count_skills, extract_skills, implied_by, or_alternatives


def test_split_chunks_strips_bullets_and_short_fragments():
    text = "SKILLS\n- Built a REST API in Flask serving 1k users.\n• Python\n\n* Led a team of four engineers."
    chunks = split_chunks(text)
    assert chunks == ["Built a REST API in Flask serving 1k users.", "Led a team of four engineers."]


def test_split_chunks_splits_long_paragraph_into_sentences():
    text = "I build models. I deploy them with Docker. I monitor them in production."
    assert len(split_chunks(text)) == 3


def test_split_chunks_deduplicates():
    assert len(split_chunks("Wrote unit tests daily\nwrote unit tests daily")) == 1


def test_sections_follow_headings():
    text = (
        "Jane Doe\nSKILLS\nPython, SQL, Docker\nEXPERIENCE\n- Built data pipelines in Airflow for analytics\n"
        "EDUCATION\nB.S. Computer Science, Example University"
    )
    sec = split_sections(text, skill_counter=count_skills)
    assert sec["skills"] == ["Python, SQL, Docker"]
    assert sec["experience"] == ["Built data pipelines in Airflow for analytics"]
    assert sec["education"] == ["B.S. Computer Science, Example University"]


def test_sections_guess_without_headings():
    text = "Python, SQL, Docker, AWS\nDeveloped dashboards for the sales team\nMaster of Science, Example University"
    sec = split_sections(text, skill_counter=count_skills)
    assert "Python, SQL, Docker, AWS" in sec["skills"]
    assert "Developed dashboards for the sales team" in sec["experience"]
    assert "Master of Science, Example University" in sec["education"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Experienced in C++ and C# development", {"C++", "C#"}),
        ("Wrote JavaScript and TypeScript", {"JavaScript", "TypeScript"}),
        ("Used Node.js and ReactJS", {"Node.js", "React"}),
        ("CI/CD pipelines with GitHub Actions", {"CI/CD", "Git"}),
        ("Deployed on k8s in AWS", {"Kubernetes", "AWS"}),
    ],
)
def test_extract_skills_aliases_and_symbols(text, expected):
    assert set(extract_skills(text)) == expected


def test_extract_skills_avoids_false_positives():
    # "Java" must not match "JavaScript", "C" must not match "C-level" or "C++",
    # "go"/"r" as plain words are not languages.
    assert "Java" not in extract_skills("JavaScript developer")
    assert "C" not in extract_skills("Reported to C-level executives using C++")
    assert extract_skills("we go to the r and d lab") == []


def test_implied_by():
    implied = implied_by(["PyTorch", "PostgreSQL"])
    assert implied["Deep learning"] == "PyTorch"
    assert implied["SQL"] == "PostgreSQL"


def test_read_document_txt_and_docx(tmp_path):
    txt = tmp_path / "r.txt"
    txt.write_text("Python developer", encoding="utf-8")
    assert read_document(txt) == "Python developer"

    d = docx.Document()
    d.add_paragraph("Data analyst skilled in SQL")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Tableau"
    table.rows[0].cells[1].text = "Excel"
    path = tmp_path / "r.docx"
    d.save(str(path))
    text = read_document(path)
    assert "Data analyst skilled in SQL" in text and "Tableau" in text and "Excel" in text


def test_or_alternatives_pairs_only_adjacent_skills():
    line = "Strong Python skills and deep learning frameworks such as PyTorch or TensorFlow."
    assert or_alternatives(line) == [("PyTorch", "TensorFlow")]
    assert or_alternatives("Tableau / Power BI dashboards") == [("Tableau", "Power BI")]
