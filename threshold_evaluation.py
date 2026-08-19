import json
import numpy as np
from itertools import combinations


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"G:\AI Face Photo Finder\fresh_labels.json"


# Thresholds to test
THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

print()
print("Loading fresh labeled dataset...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ============================================================
# REMOVE SKIP
# ============================================================

usable = [
    item
    for item in data
    if item.get("person_label") != "SKIP"
    and item.get("embedding") is not None
]


print()
print(f"Total records : {len(data)}")
print(f"Usable faces  : {len(usable)}")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

for item in usable:

    emb = np.asarray(
        item["embedding"],
        dtype=np.float32
    )

    norm = np.linalg.norm(emb)

    if len(emb) != 512 or norm == 0:

        raise ValueError(
            "Invalid embedding found: "
            f"{item.get('filename')}"
        )

    item["_embedding"] = emb / norm


# ============================================================
# BUILD PAIRS
# ============================================================

same_pairs = []
different_pairs = []


for a, b in combinations(usable, 2):

    similarity = float(
        np.dot(
            a["_embedding"],
            b["_embedding"]
        )
    )

    if (
        a["person_label"]
        ==
        b["person_label"]
    ):

        same_pairs.append(
            similarity
        )

    else:

        different_pairs.append(
            similarity
        )


same_pairs = np.asarray(
    same_pairs,
    dtype=np.float32
)

different_pairs = np.asarray(
    different_pairs,
    dtype=np.float32
)


print()
print("=" * 70)
print("PAIR DATA")
print("=" * 70)

print(
    f"SAME pairs      : "
    f"{len(same_pairs)}"
)

print(
    f"DIFFERENT pairs : "
    f"{len(different_pairs)}"
)


# ============================================================
# EVALUATE THRESHOLD
# ============================================================

def evaluate_threshold(
    threshold
):

    # SAME pair:
    # similarity >= threshold
    # means predicted SAME

    true_positive = int(
        np.sum(
            same_pairs >= threshold
        )
    )

    false_negative = int(
        np.sum(
            same_pairs < threshold
        )
    )


    # DIFFERENT pair:
    # similarity >= threshold
    # means incorrectly predicted SAME

    false_positive = int(
        np.sum(
            different_pairs >= threshold
        )
    )

    true_negative = int(
        np.sum(
            different_pairs < threshold
        )
    )


    total = (
        true_positive
        +
        true_negative
        +
        false_positive
        +
        false_negative
    )


    accuracy = (
        (true_positive + true_negative)
        / total
        if total
        else 0.0
    )


    precision = (
        true_positive
        /
        (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )


    recall = (
        true_positive
        /
        (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )


    false_positive_rate = (
        false_positive
        /
        (false_positive + true_negative)
        if (false_positive + true_negative)
        else 0.0
    )


    false_negative_rate = (
        false_negative
        /
        (false_negative + true_positive)
        if (false_negative + true_positive)
        else 0.0
    )


    specificity = (
        true_negative
        /
        (true_negative + false_positive)
        if (true_negative + false_positive)
        else 0.0
    )


    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)
        if (precision + recall)
        else 0.0
    )


    return {
        "threshold": threshold,
        "TP": true_positive,
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "FPR": false_positive_rate,
        "FNR": false_negative_rate,
        "F1": f1,
    }


# ============================================================
# RUN EVALUATION
# ============================================================

results = []

for threshold in THRESHOLDS:

    results.append(
        evaluate_threshold(
            threshold
        )
    )


# ============================================================
# PRINT TABLE
# ============================================================

print()
print("=" * 70)
print("THRESHOLD PERFORMANCE")
print("=" * 70)

print()

print(
    "Threshold | TP | FN | FP | TN | "
    "Precision | Recall | FPR | FNR | F1"
)

print("-" * 100)


for r in results:

    print(
        f"{r['threshold']:9.2f} | "
        f"{r['TP']:2d} | "
        f"{r['FN']:2d} | "
        f"{r['FP']:2d} | "
        f"{r['TN']:4d} | "
        f"{r['precision']:.4f}    | "
        f"{r['recall']:.4f} | "
        f"{r['FPR']:.4f} | "
        f"{r['FNR']:.4f} | "
        f"{r['F1']:.4f}"
    )


# ============================================================
# BEST BY F1
# ============================================================

best_f1 = max(
    results,
    key=lambda x: x["F1"]
)


print()
print("=" * 70)
print("BEST THRESHOLD BY F1")
print("=" * 70)

print(
    f"Threshold   : "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"F1          : "
    f"{best_f1['F1']:.4f}"
)

print(
    f"Precision   : "
    f"{best_f1['precision']:.4f}"
)

print(
    f"Recall      : "
    f"{best_f1['recall']:.4f}"
)

print(
    f"FPR         : "
    f"{best_f1['FPR']:.4f}"
)

print(
    f"FNR         : "
    f"{best_f1['FNR']:.4f}"
)


# ============================================================
# BEST HIGH-PRECISION THRESHOLD
# ============================================================

high_precision = [
    r
    for r in results
    if r["precision"] >= 0.99
]


if high_precision:

    best_high_precision = max(
        high_precision,
        key=lambda x: x["recall"]
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD WITH PRECISION >= 99%")
    print("=" * 70)

    print(
        f"Threshold   : "
        f"{best_high_precision['threshold']:.2f}"
    )

    print(
        f"Precision   : "
        f"{best_high_precision['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{best_high_precision['recall']:.4f}"
    )

    print(
        f"FPR         : "
        f"{best_high_precision['FPR']:.4f}"
    )

    print(
        f"FNR         : "
        f"{best_high_precision['FNR']:.4f}"
    )


# ============================================================
# BEST HIGH-RECALL THRESHOLD
# ============================================================

high_recall = [
    r
    for r in results
    if r["recall"] >= 0.99
]


if high_recall:

    best_high_recall = min(
        high_recall,
        key=lambda x: x["FPR"]
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD WITH RECALL >= 99%")
    print("=" * 70)

    print(
        f"Threshold   : "
        f"{best_high_recall['threshold']:.2f}"
    )

    print(
        f"Precision   : "
        f"{best_high_recall['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{best_high_recall['recall']:.4f}"
    )

    print(
        f"FPR         : "
        f"{best_high_recall['FPR']:.4f}"
    )

    print(
        f"FNR         : "
        f"{best_high_recall['FNR']:.4f}"
    )


# ============================================================
# ACTUAL RANGE-BASED BOUNDARIES
# ============================================================

same_min = float(
    np.min(same_pairs)
)

different_max = float(
    np.max(different_pairs)
)

safe_gap_midpoint = (
    same_min + different_max
) / 2.0


print()
print("=" * 70)
print("OBSERVED DATA BOUNDARIES")
print("=" * 70)

print(
    f"Highest DIFFERENT similarity : "
    f"{different_max:.4f}"
)

print(
    f"Lowest SAME similarity       : "
    f"{same_min:.4f}"
)

print(
    f"Midpoint                     : "
    f"{safe_gap_midpoint:.4f}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "These boundaries are based ONLY on"
)

print(
    "the current labeled dataset."
)

print(
    "They are NOT production thresholds yet."
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "usable_faces": len(usable),
    "same_pairs": len(same_pairs),
    "different_pairs": len(different_pairs),
    "same_min": same_min,
    "different_max": different_max,
    "observed_gap_midpoint": safe_gap_midpoint,
    "threshold_results": results,
}


with open(
    r"G:\AI Face Photo Finder\threshold_evaluation.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


print()
print("=" * 70)
print("Evaluation completed.")
print("=" * 70)

print()
print(
    "Saved:"
)

print(
    r"G:\AI Face Photo Finder\threshold_evaluation.json"
)