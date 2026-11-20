"""Turn resumes and job descriptions into clean text and small chunks.

* ``read_document`` reads PDF / DOCX / TXT files into plain text.
* ``split_chunks`` cuts text into short pieces (bullets / sentences). Short
  pieces embed much better than a whole 2-page resume, and they let us show
  *which line* of the resume supports each job requirement.
* ``split_sections`` groups resume lines into skills / experience / education.
"""

from __future__ import annotations

import re
from pathlib import Path

SECTIONS = ("skills", "experience", "education")

# Headings that start a section. A line counts as a heading when it is short
# and matches one of these (e.g. "TECHNICAL SKILLS", "Work Experience:").
_HEADINGS = {
    "skills": r"(technical\s+)?skills|core\s+competencies|competencies|technologies|tech(nical)?\s+stack|tools|expertise|skill\s+set",
    "experience": r"(work|professional|relevant|employment)?\s*(experience|history)|employment|projects|work",
    "education": r"education|academic(s|\s+background)?|qualifications|certifications?|courses|training",
}
_OTHER_HEADINGS = r"summary|profile|objective|about(\s+me)?|contact|interests|hobbies|references|awards|languages|publications"

# Words that mark a line as being about education when there are no headings.
_EDU_WORDS = re.compile(
    r"\b(bachelor|master|b\.?\s?s\.?c?|m\.?\s?s\.?c?|b\.?\s?tech|m\.?\s?tech|b\.?e\.|ph\.?d|mba|degree|diploma|"
    r"university|college|institute|gpa|graduated|graduation|coursework|certificat\w*)\b",
    re.IGNORECASE,
)

_BULLET = re.compile(r"^\s*([\-•●▪‣⁃\*·>o]|\d+[.)])\s+")
# A sentence ends at . ! ? ; after a lowercase word, digit or bracket, followed by a
# capital letter/digit. "B.S. Computer" and "e.g. Python" are not sentence ends.
_SENTENCE_END = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[a-z0-9%)][.!?;])\s+(?=[A-Z0-9])")


# --------------------------------------------------------------------------- #
# Reading files
# --------------------------------------------------------------------------- #
def read_document(path: str | Path) -> str:
    """Read a .pdf, .docx or plain-text file and return its text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx  # the python-docx package

        doc = docx.Document(str(path))
        lines = [p.text for p in doc.paragraphs]
        # Resumes often put skills in tables, so read table cells too.
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(lines)
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def clean_line(line: str) -> str:
    """Strip bullets and squeeze whitespace."""
    line = _BULLET.sub("", line)
    return re.sub(r"\s+", " ", line).strip(" \t-|:")


def _split_long(text: str, max_words: int) -> list[str]:
    """Split text into sentences, then cut any still-too-long sentence into windows."""
    pieces = []
    for sent in _SENTENCE_END.split(text):
        words = sent.split()
        for start in range(0, len(words), max_words):
            pieces.append(" ".join(words[start : start + max_words]))
    return pieces


def split_chunks(text: str, min_words: int = 3, max_words: int = 40, max_chunks: int = 200) -> list[str]:
    """Cut a document into short, meaningful chunks (bullets or sentences).

    Very short fragments (fewer than ``min_words`` words, e.g. "Python" or a
    phone number) are dropped because they carry little meaning on their own;
    skills are handled separately by the skill extractor.
    """
    chunks: list[str] = []
    seen: set[str] = set()
    for raw in text.replace("\r", "\n").split("\n"):
        line = clean_line(raw)
        if not line:
            continue
        for piece in _split_long(line, max_words):
            piece = piece.strip()
            key = piece.lower()
            if len(piece.split()) >= min_words and key not in seen:
                seen.add(key)
                chunks.append(piece)
    return chunks[:max_chunks]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _heading_section(line: str) -> str | None:
    """Return the section a heading line starts ('other' for unknown headings), or None."""
    text = clean_line(line).lower().rstrip(":")
    if not text or len(text.split()) > 4:
        return None
    for section, pattern in _HEADINGS.items():
        if re.fullmatch(pattern, text):
            return section
    if re.fullmatch(_OTHER_HEADINGS, text):
        return "other"
    return None


def is_education_line(line: str) -> bool:
    return bool(_EDU_WORDS.search(line))


def split_sections(text: str, skill_counter=None) -> dict[str, list[str]]:
    """Group the chunks of a resume into skills / experience / education / other.

    If the resume has headings ("Skills", "Experience", ...) we follow them.
    Otherwise (e.g. text pasted as one paragraph) each chunk is classified on
    its own: education words -> education, mostly a list of skills -> skills,
    everything else -> experience.

    ``skill_counter`` is an optional function ``str -> int`` that counts known
    skills in a line (passed in to avoid a circular import).
    """
    sections: dict[str, list[str]] = {s: [] for s in (*SECTIONS, "other")}
    current: str | None = None  # section of the last heading seen

    for raw in text.replace("\r", "\n").split("\n"):
        heading = _heading_section(raw)
        if heading:
            current = heading
            continue
        for chunk in split_chunks(raw, min_words=1):
            if current is not None:
                sections[current].append(chunk)
            else:
                # Before any heading (or no headings at all): guess per chunk.
                sections[_guess_section(chunk, skill_counter)].append(chunk)
    return sections


def _guess_section(chunk: str, skill_counter=None) -> str:
    if is_education_line(chunk):
        return "education"
    if skill_counter is not None:
        n_words = len(chunk.split())
        n_skills = skill_counter(chunk)
        # A line like "Python, SQL, Docker, AWS" is a skills line.
        if n_skills >= 3 or (n_skills >= 1 and n_words <= 6 and "," in chunk):
            return "skills"
    if len(chunk.split()) < 3:
        return "other"
    return "experience"
