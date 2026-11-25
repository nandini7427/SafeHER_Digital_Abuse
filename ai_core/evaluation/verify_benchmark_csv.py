"""Audits and mathematically verifies every row in per_image_evaluation_results.csv.

Computes exact confusion matrices, FPR, FNR, balanced accuracy, discrete score AUC,
continuous variance AUC, and 95% Wilson score confidence intervals.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p = k / n
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    margin = (z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)


def compute_roc_auc(y_true: List[bool], y_scores: List[float]) -> float | None:
    n_pos = sum(1 for yt in y_true if yt)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    combined = sorted(enumerate(y_scores), key=lambda x: x[1])
    ranks = [0.0] * len(y_scores)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][1] == combined[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[combined[k][0]] = avg_rank
        i = j

    sum_pos_ranks = sum(ranks[idx] for idx, yt in enumerate(y_true) if yt)
    u_statistic = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u_statistic / (n_pos * n_neg)
    return round(float(auc), 4)


def main():
    csv_path = Path(__file__).resolve().parent / "per_image_evaluation_results.csv"
    assert csv_path.exists(), f"CSV path {csv_path} does not exist"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"Loaded {len(reader)} evaluation rows from {csv_path.name}")

    groups: Dict[str, List[dict]] = {}
    for r in reader:
        groups.setdefault(r["source_dataset"], []).append(r)

    print("\n" + "=" * 95)
    print(f"{'Dataset':<26} | {'N':<3} | {'TP':<2} {'FP':<2} {'TN':<2} {'FN':<2} | {'Bal Acc':<7} | {'Disc AUC':<8} | {'Cont AUC':<8} | {'FPR (95% CI)':<16} | {'FNR':<6}")
    print("=" * 95)

    for ds_name, rows in groups.items():
        y_true = [r["is_manipulated"] == "True" for r in rows]
        y_pred = [r["is_predicted_manipulated"] == "True" for r in rows]
        raw_scores = [float(r["raw_model_score"]) for r in rows]
        raw_vars = [float(r["raw_ela_variance"]) for r in rows]

        n_pos = sum(1 for yt in y_true if yt)
        n_neg = sum(1 for yt in y_true if not yt)

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

        fpr = fp / n_neg if n_neg > 0 else 0.0
        ci_low, ci_high = compute_wilson_ci(fp, n_neg)
        fpr_str = f"{fpr*100:4.1f}% [{ci_low*100:4.1f}%, {ci_high*100:4.1f}%]"

        if n_pos > 0 and n_neg > 0:
            tpr = tp / n_pos
            tnr = tn / n_neg
            bal_acc = (tpr + tnr) / 2.0
            bal_str = f"{bal_acc*100:5.1f}%"
            fnr_str = f"{(fn/n_pos)*100:5.1f}%"
            disc_auc = compute_roc_auc(y_true, raw_scores)
            cont_auc = compute_roc_auc(y_true, raw_vars)
            disc_auc_str = f"{disc_auc:.4f}" if disc_auc is not None else "  N/A"
            cont_auc_str = f"{cont_auc:.4f}" if cont_auc is not None else "  N/A"
        else:
            bal_str = "  N/A"
            fnr_str = " N/A"
            disc_auc_str = "  N/A"
            cont_auc_str = "  N/A"

        print(f"{ds_name:<26} | {len(rows):<3} | {tp:<2} {fp:<2} {tn:<2} {fn:<2} | {bal_str:<7} | {disc_auc_str:<8} | {cont_auc_str:<8} | {fpr_str:<16} | {fnr_str:<6}")

    print("=" * 95)


if __name__ == "__main__":
    main()
