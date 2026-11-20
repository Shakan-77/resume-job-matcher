"""Resume <-> Job Description matcher built on sentence embeddings.

Importing this package first (before transformers / sentence_transformers /
datasets / gradio) keeps every file the project writes inside the project
folder: downloaded models and datasets, Gradio uploads and temp files.
These folders are gitignored. A user-chosen HF_HOME / GRADIO_TEMP_DIR wins.
"""

import os
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
_TMP_DIR = PROJECT_DIR / ".tmp"
_TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(_TMP_DIR / "gradio"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_DIR / "matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without symlinks: harmless
tempfile.tempdir = str(_TMP_DIR)

RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
EXAMPLES_DIR = PROJECT_DIR / "examples"
