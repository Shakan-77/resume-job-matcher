"""Fine-tune the MiniLM embedding model on labelled resume/JD pairs.

Off-the-shelf embedding models know general meaning ("Python" ~ "programming")
but not what recruiters consider a *fit*. Here we show the model ~6k labelled
pairs and nudge it so that Good Fit pairs end up closer together than No Fit
pairs (CoSENT loss: it only cares that the *ranking* of similarities matches
the labels No Fit < Potential Fit < Good Fit).

Hyper-parameters are fixed up front (not tuned on the test set).
Usage:  python finetune.py            (~5-10 min on a laptop GPU)
"""

from __future__ import annotations

import argparse
import subprocess
import time

from matcher import PROJECT_DIR  # isort: skip  (sets HF_HOME first)

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CoSENTLoss

from matcher.data import load_pairs
from matcher.embedder import FINETUNED_DIR, MODELS
from matcher.scoring import job_doc_text, resume_doc_text

LABEL_TO_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}  # No / Potential / Good Fit -> target similarity


def free_gpu_mib() -> int:
    """Free memory on GPU 0 as reported by nvidia-smi (the GPU is shared with other jobs)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.split()[0])
    except Exception:
        return 0


def wait_for_gpu(min_free_mib: int, timeout_s: int) -> bool:
    """Wait until the shared GPU has enough free memory. Returns False on timeout."""
    start = time.time()
    while free_gpu_mib() < min_free_mib:
        if time.time() - start > timeout_s:
            return False
        print(f"Waiting for GPU memory (free {free_gpu_mib()} MiB < {min_free_mib} MiB)...", flush=True)
        time.sleep(30)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--gpu-wait-min", type=int, default=30, help="minutes to wait for a free GPU before using the CPU")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available() and wait_for_gpu(1500, timeout_s=args.gpu_wait_min * 60)
    device = "cuda" if use_gpu else "cpu"
    print(f"Training on {device}", flush=True)

    train = load_pairs("train")
    dataset = Dataset.from_dict({
        "resume": [resume_doc_text(t) for t in train["resume"]],
        "job": [job_doc_text(t) for t in train["job"]],
        "score": [LABEL_TO_SCORE[y] for y in train["y"]],
    }).shuffle(seed=42)

    model = SentenceTransformer(MODELS["minilm"], device=device)
    model.max_seq_length = args.max_seq_length  # longer than the default 256: resumes are long
    loss = CoSENTLoss(model)

    batch = args.batch_size
    while True:
        try:
            training_args = SentenceTransformerTrainingArguments(
                output_dir=str(PROJECT_DIR / ".tmp" / "finetune-checkpoints"),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=batch,
                gradient_accumulation_steps=max(1, args.batch_size // batch),  # keep the effective batch size
                learning_rate=2e-5,
                warmup_ratio=0.1,
                fp16=use_gpu,
                use_cpu=not use_gpu,
                save_strategy="no",
                logging_steps=50,
                report_to="none",
                seed=42,
            )
            trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=dataset, loss=loss)
            start = time.time()
            trainer.train()
            break
        except torch.OutOfMemoryError:
            # Another process grabbed the GPU: free what we hold, halve the batch, retry.
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch //= 2
            print(f"CUDA out of memory - retrying with per-device batch size {batch}")
            time.sleep(20)

    FINETUNED_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FINETUNED_DIR))
    print(f"Saved fine-tuned model to {FINETUNED_DIR} after {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
