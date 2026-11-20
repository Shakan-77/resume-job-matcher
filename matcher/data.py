"""Load the labelled resume / job-description dataset used for evaluation.

Source: https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit
8,000 (resume, job description) pairs labelled "No Fit", "Potential Fit" or
"Good Fit" (6,241 train / 1,759 test). The test split uses job descriptions
that never appear in train, so fine-tuning on train can't memorise them.
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

DATASET = "cnamuangtoun/resume-job-description-fit"
LABELS = ["No Fit", "Potential Fit", "Good Fit"]  # ordered from worst to best


def load_pairs(split: str) -> pd.DataFrame:
    """DataFrame with columns: resume, job, label (str), y (0/1/2 = No/Potential/Good)."""
    df = load_dataset(DATASET, split=split).to_pandas()
    df = df.rename(columns={"resume_text": "resume", "job_description_text": "job"})
    df["y"] = df["label"].map(LABELS.index)
    return df[["resume", "job", "label", "y"]]
