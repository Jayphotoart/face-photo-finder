import psycopg2
import numpy as np


DB_CONFIG = {
    "host": "localhost",
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto",
    "port": 5432
}


def main():

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT similarity, result
        FROM verification_results
        WHERE result IN ('Y', 'N')
        ORDER BY similarity DESC
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        print("No verification results found.")
        return

    similarities = np.array([float(r[0]) for r in rows])
    actual = np.array([1 if r[1] == 'Y' else 0 for r in rows])

    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(f"Total pairs: {len(rows)}")
    print(f"Same person : {np.sum(actual == 1)}")
    print(f"Different   : {np.sum(actual == 0)}")

    print()
    print("-" * 70)
    print("THRESHOLD RESULTS")
    print("-" * 70)

    best = None

    for threshold in np.arange(0.20, 0.91, 0.01):

        predicted = (similarities >= threshold).astype(int)

        TP = np.sum((predicted == 1) & (actual == 1))
        TN = np.sum((predicted == 0) & (actual == 0))
        FP = np.sum((predicted == 1) & (actual == 0))
        FN = np.sum((predicted == 0) & (actual == 1))

        accuracy = (TP + TN) / len(actual)

        precision = (
            TP / (TP + FP)
            if (TP + FP) > 0
            else 0
        )

        recall = (
            TP / (TP + FN)
            if (TP + FN) > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if best is None or f1 > best["f1"]:
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN
            }

        if threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
            print(
                f"Threshold={threshold:.2f} | "
                f"Acc={accuracy:.3f} | "
                f"Precision={precision:.3f} | "
                f"Recall={recall:.3f} | "
                f"F1={f1:.3f} | "
                f"TP={TP} TN={TN} FP={FP} FN={FN}"
            )

    print()
    print("=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    print(f"Threshold : {best['threshold']:.2f}")
    print(f"Accuracy  : {best['accuracy']:.3f}")
    print(f"Precision : {best['precision']:.3f}")
    print(f"Recall    : {best['recall']:.3f}")
    print(f"F1 Score  : {best['f1']:.3f}")

    print()
    print("Confusion Matrix:")
    print(f"TP = {best['TP']}")
    print(f"TN = {best['TN']}")
    print(f"FP = {best['FP']}")
    print(f"FN = {best['FN']}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()