import subprocess
import re
import csv
import sys
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

FACE_SEARCH_SCRIPT = "face_search.py"

TEST_IMAGES = {
    1: r"test_images\test1.jpg",
    2: r"test_images\test2.jpg",
    3: r"test_images\test3.jpg",
    4: r"test_images\test4.jpg",
    5: r"test_images\test5.jpg",
    6: r"test_images\test6.jpg",
    7: r"test_images\test7.jpg",
    8: r"test_images\test8.jpg",
    9: r"test_images\test9.jpg",
}



# ============================================================
# FIXED GROUND TRUTH
# ============================================================

EXPECTED = {
    1: ["D"],
    2: ["C", "D", "B", "A"],
    3: ["B"],
    4: ["C", "D"],
    5: ["B", "C"],
    6: ["A", "C"],
    7: ["B", "C", "A", "D"],
    8: ["A", "B", "D", "C"],
    9: ["B", "C", "A", "D"],
}


RESULT_FILE = "benchmark_results.csv"


# ============================================================
# RUN ACTUAL FACE SEARCH
# ============================================================

def run_face_search(image_path):

    print()
    print("-" * 70)
    print("Running:", image_path)
    print("-" * 70)

    try:

        result = subprocess.run(
            [
                sys.executable,
                FACE_SEARCH_SCRIPT,
                image_path
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        output = result.stdout + "\n" + result.stderr

        # Show actual face_search.py output
        print(output)

        return {
            "success": result.returncode == 0,
            "output": output,
            "error": ""
        }

    except Exception as e:

        print("SUBPROCESS ERROR:")
        print(e)

        return {
            "success": False,
            "output": "",
            "error": str(e)
        }


# ============================================================
# PARSE PERSON RANKING
# ============================================================

def parse_person_ranking(output):

    persons = []

    inside_person_section = False

    for line in output.splitlines():

        line = line.strip()

        if line == "PERSON LEVEL AGGREGATION":

            inside_person_section = True
            continue

        if not inside_person_section:
            continue

        match = re.match(
            r"^\d+\.\s+PERSON\s+([A-D])$",
            line
        )

        if match:

            person = match.group(1)

            persons.append(person)

    return persons


# ============================================================
# PARSE PERSON SCORES
# ============================================================

def parse_person_scores(output):

    scores = {}

    current_person = None

    inside_person_section = False

    for line in output.splitlines():

        line = line.strip()

        if line == "PERSON LEVEL AGGREGATION":

            inside_person_section = True
            continue

        if not inside_person_section:
            continue

        match_person = re.match(
            r"^\d+\.\s+PERSON\s+([A-D])$",
            line
        )

        if match_person:

            current_person = match_person.group(1)
            continue

        if current_person:

            match_score = re.match(
                r"^Person Score\s*:\s*([0-9.]+)",
                line
            )

            if match_score:

                scores[current_person] = float(
                    match_score.group(1)
                )

                current_person = None

    return scores


# ============================================================
# EVALUATE TEST
# ============================================================

def evaluate(test_number, actual):

    expected = EXPECTED[test_number]

    # --------------------------------------------------------
    # Top-1
    # --------------------------------------------------------

    top1 = (
        len(actual) > 0
        and actual[0] == expected[0]
    )

    # --------------------------------------------------------
    # Exact ranking
    # --------------------------------------------------------

    exact = actual == expected

    # --------------------------------------------------------
    # Expected persons found
    # --------------------------------------------------------

    expected_found = all(
        person in actual
        for person in expected
    )

    # --------------------------------------------------------
    # Position accuracy
    # --------------------------------------------------------

    correct_positions = 0

    for i in range(
        min(len(expected), len(actual))
    ):

        if expected[i] == actual[i]:

            correct_positions += 1

    if len(expected) > 0:

        ranking_accuracy = (
            correct_positions
            / len(expected)
            * 100
        )

    else:

        ranking_accuracy = 0.0

    return {
        "test": test_number,
        "expected": ">".join(expected),
        "actual": ">".join(actual),
        "top1": "PASS" if top1 else "FAIL",
        "all_expected_found": (
            "PASS"
            if expected_found
            else "FAIL"
        ),
        "exact_ranking": (
            "PASS"
            if exact
            else "FAIL"
        ),
        "ranking_accuracy": round(
            ranking_accuracy,
            1
        )
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test",
                "expected",
                "actual",
                "top1",
                "all_expected_found",
                "exact_ranking",
                "ranking_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI FACE PHOTO FINDER")
    print("AUTOMATED 9-TEST BENCHMARK")
    print("=" * 70)

    print()
    print("Ground truth is FIXED.")
    print("Actual results come directly from face_search.py.")
    print()

    results = []

    # --------------------------------------------------------
    # RUN ALL 9 TESTS
    # --------------------------------------------------------

    for test_number in range(1, 10):

        image_path = TEST_IMAGES[test_number]

        run_result = run_face_search(
            image_path
        )

        if not run_result["success"]:

            print()
            print(
                f"Test {test_number}: ERROR"
            )

            results.append({
                "test": test_number,
                "expected": ">".join(
                    EXPECTED[test_number]
                ),
                "actual": "ERROR",
                "top1": "ERROR",
                "all_expected_found": "ERROR",
                "exact_ranking": "ERROR",
                "ranking_accuracy": 0.0
            })

            continue

        output = run_result["output"]

        actual = parse_person_ranking(
            output
        )

        scores = parse_person_scores(
            output
        )

        evaluation = evaluate(
            test_number,
            actual
        )

        results.append(
            evaluation
        )

        print()
        print(
            f"Test {test_number}"
        )

        print(
            f"Expected : "
            f"{evaluation['expected']}"
        )

        print(
            f"Actual   : "
            f"{evaluation['actual']}"
        )

        print(
            f"Top-1    : "
            f"{evaluation['top1']}"
        )

        print(
            f"Exact    : "
            f"{evaluation['exact_ranking']}"
        )

        print(
            f"Ranking  : "
            f"{evaluation['ranking_accuracy']}%"
        )

        if scores:

            print("Scores:")

            for person, score in scores.items():

                print(
                    f"   Person {person}: "
                    f"{score:.4f}"
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(results)

    top1_pass = sum(
        r["top1"] == "PASS"
        for r in results
    )

    exact_pass = sum(
        r["exact_ranking"] == "PASS"
        for r in results
    )

    expected_found_pass = sum(
        r["all_expected_found"] == "PASS"
        for r in results
    )

    average_ranking_accuracy = (
        sum(
            r["ranking_accuracy"]
            for r in results
        )
        / total
    )

    print()
    print("=" * 70)
    print("FINAL BENCHMARK RESULT")
    print("=" * 70)

    print(
        f"Top-1 Accuracy       : "
        f"{top1_pass}/{total} "
        f"({top1_pass / total * 100:.1f}%)"
    )

    print(
        f"Expected Persons     : "
        f"{expected_found_pass}/{total} "
        f"({expected_found_pass / total * 100:.1f}%)"
    )

    print(
        f"Exact Ranking        : "
        f"{exact_pass}/{total} "
        f"({exact_pass / total * 100:.1f}%)"
    )

    print(
        f"Average Rank Score   : "
        f"{average_ranking_accuracy:.1f}%"
    )

    save_results(results)

    print()
    print(
        f"CSV saved: {RESULT_FILE}"
    )

    print(
        "Completed:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print("=" * 70)


if __name__ == "__main__":

    main()