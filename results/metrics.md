# Evaluation: does the score separate good fits from poor ones?

Dataset: [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit), test split (1759 pairs: 857 No Fit, 444 Potential Fit, 458 Good Fit). Test job descriptions never appear in the training split.

| Method | ROC-AUC Good vs No | ROC-AUC Fit vs No | Spearman |
|---|---:|---:|---:|
| TF-IDF cosine (baseline) | 0.602 | 0.586 | 0.153 |
| Skill overlap (vocabulary only) | 0.557 | 0.556 | 0.091 |
| minilm whole-doc | 0.539 | 0.572 | 0.088 |
| minilm matcher | 0.519 | 0.559 | 0.061 |
| minilm matcher (learned weights) | 0.516 | 0.559 | 0.059 |
| bge-small whole-doc | 0.600 | 0.619 | 0.175 |
| bge-small matcher | 0.549 | 0.580 | 0.101 |
| bge-small matcher (learned weights) | 0.580 | 0.606 | 0.148 |
| finetuned whole-doc | **0.800** | 0.770 | 0.470 |
| finetuned matcher | 0.731 | 0.726 | 0.371 |

*ROC-AUC*: chance that a random better-fitting pair gets a higher score than a random worse one (0.5 = coin flip, 1.0 = perfect). *Spearman*: rank correlation with the 3-level label.

## Learned component weights (logistic regression on train, per 1 point of 0-100 score)

- minilm: requirements +0.0197, skills +0.0028, document +0.0149
- bge-small: requirements +0.0158, skills +0.0022, document +0.0281

![ROC-AUC by method](auc_comparison.png)

![Score by label](score_by_label.png)
