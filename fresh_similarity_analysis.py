import json
import numpy as np
from itertools import combinations
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"G:\AI Face Photo Finder\fresh_labels.json"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FRESH FACE SIMILARITY ANALYSIS")
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
# BASIC INTEGRITY
# ============================================================

print()
print("=" * 70)
print("DATASET INTEGRITY")
print("=" * 70)

dimensions = Counter(
    len(item["embedding"])
    for item in usable
)

print(
    f"Embedding dimensions: {dimensions}"
)


bad_embeddings = []

for item in usable:

    emb = np.asarray(
        item["embedding"],
        dtype=np.float32
    )

    norm = np.linalg.norm(emb)

    if len(emb) != 512 or norm == 0:

        bad_embeddings.append(
            item
        )


print(
    f"Bad/zero embeddings : "
    f"{len(bad_embeddings)}"
)


# ============================================================
# LABEL COUNTS
# ============================================================

label_counts = Counter(
    item["person_label"]
    for item in usable
)

print()
print("=" * 70)
print("LABELED FACE COUNTS")
print("=" * 70)

for label in sorted(label_counts):

    print(
        f"Person {label}: "
        f"{label_counts[label]}"
    )


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

for item in usable:

    emb = np.asarray(
        item["embedding"],
        dtype=np.float32
    )

    norm = np.linalg.norm(emb)

    item["_embedding"] = emb / norm


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
    )


# ============================================================
# BUILD PAIRS
# ============================================================

same_pairs = []
different_pairs = []


for a, b in combinations(
    usable,
    2
):

    similarity = cosine_similarity(
        a["_embedding"],
        b["_embedding"]
    )

    pair = {
        "similarity": similarity,
        "a": a,
        "b": b
    }


    if (
        a["person_label"]
        ==
        b["person_label"]
    ):

        same_pairs.append(
            pair
        )

    else:

        different_pairs.append(
            pair
        )


# ============================================================
# PAIR SUMMARY
# ============================================================

print()
print("=" * 70)
print("PAIR SUMMARY")
print("=" * 70)

print(
    f"Total pairs     : "
    f"{len(same_pairs) + len(different_pairs)}"
)

print(
    f"SAME pairs      : "
    f"{len(same_pairs)}"
)

print(
    f"DIFFERENT pairs : "
    f"{len(different_pairs)}"
)


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    title,
    pairs
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not pairs:

        print("No pairs.")
        return

    values = np.array(
        [
            p["similarity"]
            for p in pairs
        ],
        dtype=np.float32
    )

    print(
        f"Pair count : {len(values)}"
    )

    print(
        f"Minimum    : "
        f"{np.min(values):.4f}"
    )

    print(
        f"Maximum    : "
        f"{np.max(values):.4f}"
    )

    print(
        f"Average    : "
        f"{np.mean(values):.4f}"
    )

    print(
        f"Median     : "
        f"{np.median(values):.4f}"
    )

    print(
        f"P05        : "
        f"{np.percentile(values, 5):.4f}"
    )

    print(
        f"P10        : "
        f"{np.percentile(values, 10):.4f}"
    )

    print(
        f"P25        : "
        f"{np.percentile(values, 25):.4f}"
    )

    print(
        f"P50        : "
        f"{np.percentile(values, 50):.4f}"
    )

    print(
        f"P75        : "
        f"{np.percentile(values, 75):.4f}"
    )

    print(
        f"P90        : "
        f"{np.percentile(values, 90):.4f}"
    )

    print(
        f"P95        : "
        f"{np.percentile(values, 95):.4f}"
    )


# ============================================================
# PRINT STATISTICS
# ============================================================

print_statistics(
    "SAME-PERSON SIMILARITY",
    same_pairs
)

print_statistics(
    "DIFFERENT-PERSON SIMILARITY",
    different_pairs
)


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

print()
print("=" * 70)
print("OVERLAP ANALYSIS")
print("=" * 70)


same_values = np.array(
    [
        p["similarity"]
        for p in same_pairs
    ]
)

different_values = np.array(
    [
        p["similarity"]
        for p in different_pairs
    ]
)


same_min = np.min(
    same_values
)

same_max = np.max(
    same_values
)

different_min = np.min(
    different_values
)

different_max = np.max(
    different_values
)


print(
    f"SAME minimum       : "
    f"{same_min:.4f}"
)

print(
    f"SAME maximum       : "
    f"{same_max:.4f}"
)

print(
    f"DIFFERENT minimum  : "
    f"{different_min:.4f}"
)

print(
    f"DIFFERENT maximum  : "
    f"{different_max:.4f}"
)


overlap_low = max(
    same_min,
    different_min
)

overlap_high = min(
    same_max,
    different_max
)


if overlap_low <= overlap_high:

    print()
    print(
        "WARNING: Similarity ranges overlap."
    )

    print(
        f"Overlap range: "
        f"{overlap_low:.4f} - "
        f"{overlap_high:.4f}"
    )

    print()
    print(
        "A single threshold will NOT perfectly"
    )

    print(
        "separate SAME and DIFFERENT persons."
    )

else:

    print()
    print(
        "No range overlap detected."
    )


# ============================================================
# TOP DIFFERENT-PERSON PAIRS
# ============================================================

print()
print("=" * 70)
print("TOP 20 DIFFERENT-PERSON PAIRS")
print("=" * 70)


top_different = sorted(
    different_pairs,
    key=lambda x: x["similarity"],
    reverse=True
)[:20]


for index, pair in enumerate(
    top_different,
    start=1
):

    a = pair["a"]
    b = pair["b"]

    print()
    print(
        f"#{index} "
        f"Similarity: "
        f"{pair['similarity']:.6f}"
    )

    print(
        f"   {a['person_label']} | "
        f"{a['filename']} | "
        f"Face={a['face_index']}"
    )

    print(
        f"   {b['person_label']} | "
        f"{b['filename']} | "
        f"Face={b['face_index']}"
    )


# ============================================================
# BOTTOM SAME-PERSON PAIRS
# ============================================================

print()
print("=" * 70)
print("BOTTOM 20 SAME-PERSON PAIRS")
print("=" * 70)


bottom_same = sorted(
    same_pairs,
    key=lambda x: x["similarity"]
)[:20]


for index, pair in enumerate(
    bottom_same,
    start=1
):

    a = pair["a"]
    b = pair["b"]

    print()
    print(
        f"#{index} "
        f"Similarity: "
        f"{pair['similarity']:.6f}"
    )

    print(
        f"   Person {a['person_label']}"
    )

    print(
        f"   {a['filename']} | "
        f"Face={a['face_index']}"
    )

    print(
        f"   {b['filename']} | "
        f"Face={b['face_index']}"
    )


# ============================================================
# PER-PERSON SAME-PERSON STATISTICS
# ============================================================

print()
print("=" * 70)
print("PER-PERSON SAME-PERSON ANALYSIS")
print("=" * 70)


for label in sorted(label_counts):

    person_pairs = [
        p
        for p in same_pairs
        if p["a"]["person_label"] == label
    ]

    if not person_pairs:

        continue

    values = np.array(
        [
            p["similarity"]
            for p in person_pairs
        ]
    )

    print()
    print(
        f"PERSON {label}"
    )

    print(
        f"   Pair count : "
        f"{len(values)}"
    )

    print(
        f"   Min        : "
        f"{np.min(values):.4f}"
    )

    print(
        f"   Max        : "
        f"{np.max(values):.4f}"
    )

    print(
        f"   Average    : "
        f"{np.mean(values):.4f}"
    )

    print(
        f"   Median     : "
        f"{np.median(values):.4f}"
    )

    print(
        f"   P10        : "
        f"{np.percentile(values, 10):.4f}"
    )

    print(
        f"   P25        : "
        f"{np.percentile(values, 25):.4f}"
    )


# ============================================================
# SAVE ANALYSIS SUMMARY
# ============================================================

summary = {

    "usable_faces": len(usable),

    "label_counts": dict(
        label_counts
    ),

    "total_pairs":
        len(same_pairs)
        +
        len(different_pairs),

    "same_pairs":
        len(same_pairs),

    "different_pairs":
        len(different_pairs),

    "same_statistics": {

        "min":
            float(np.min(same_values)),

        "max":
            float(np.max(same_values)),

        "average":
            float(np.mean(same_values)),

        "median":
            float(np.median(same_values)),

        "p05":
            float(np.percentile(same_values, 5)),

        "p10":
            float(np.percentile(same_values, 10)),

        "p25":
            float(np.percentile(same_values, 25)),

        "p50":
            float(np.percentile(same_values, 50)),

        "p75":
            float(np.percentile(same_values, 75)),

        "p90":
            float(np.percentile(same_values, 90)),

        "p95":
            float(np.percentile(same_values, 95))
    },

    "different_statistics": {

        "min":
            float(np.min(different_values)),

        "max":
            float(np.max(different_values)),

        "average":
            float(np.mean(different_values)),

        "median":
            float(np.median(different_values)),

        "p05":
            float(np.percentile(different_values, 5)),

        "p10":
            float(np.percentile(different_values, 10)),

        "p25":
            float(np.percentile(different_values, 25)),

        "p50":
            float(np.percentile(different_values, 50)),

        "p75":
            float(np.percentile(different_values, 75)),

        "p90":
            float(np.percentile(different_values, 90)),

        "p95":
            float(np.percentile(different_values, 95))
    },

    "overlap": {

        "low":
            float(overlap_low),

        "high":
            float(overlap_high)
    }
}


with open(
    r"G:\AI Face Photo Finder\fresh_similarity_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("Analysis completed.")
print("=" * 70)

print()
print(
    "Summary saved to:"
)

print(
    r"G:\AI Face Photo Finder\fresh_similarity_summary.json"
)