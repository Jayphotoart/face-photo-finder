import sys
import cv2
import json
import numpy as np
from itertools import permutations
from insightface.app import FaceAnalysis

# ============================================================
# 1. BUILD SIMILARITY MATRIX
# ============================================================
def build_person_similarity_matrix(query_embeddings, query_face_matches):
    """
    Build: query_face -> person -> best similarity
    Only the BEST database embedding for each query-face + person combination is used.
    """
    matrix = {}
    for query_data in query_embeddings:
        query_index = query_data["query_face"]
        matrix[query_index] = {}
        for match in query_face_matches.get(query_index, []):
            person = match["person"]
            similarity = match["similarity"]
            current = matrix[query_index].get(person)
            if current is None or similarity > current["similarity"]:
                matrix[query_index][person] = match
    return matrix


# ============================================================
# 2. GLOBAL ASSIGNMENT (WITH FALLBACK)
# ============================================================
def find_best_global_assignment(query_embeddings, query_face_matches, persons_list):
    """
    Assign query faces to different persons.
    One person can be assigned to only ONE query face.
    persons_list: ['A', 'B', 'C', 'D']
    """

    # 1. Build similarity matrix
    matrix = build_person_similarity_matrix(query_embeddings, query_face_matches)

    # Debug prints (for benchmark parsing)
    print("\n========== PERSON MATRIX ==========")
    for q, persons in matrix.items():
        print(f"Query {q}: {list(persons.keys())}")
        for p, m in persons.items():
            print(f"  {p}: {round(m['similarity'], 4)}")
    print("====================================\n")

    # 2. Separate valid / invalid queries
    valid_queries = {q: persons for q, persons in matrix.items() if persons}
    invalid_queries = [q for q, persons in matrix.items() if not persons]
    query_faces = list(valid_queries.keys())

    if not query_faces:
        return [None] * len(matrix.keys())

    # 3. Find available persons in matrix
    available_persons = set()
    for persons in valid_queries.values():
        available_persons.update(persons.keys())

    possible_persons = [p for p in persons_list if p in available_persons]

    # 4. If fewer persons than query faces -> fallback to greedy assignment
    if len(possible_persons) < len(query_faces):
        print("[WARNING] Available persons are less than query faces. Taking best individual matches with unique persons.")
        used_persons = set()
        fallback_assignment = []
        # Sort query faces by best score descending to prioritize confident matches
        for q in sorted(query_faces, key=lambda q: max(valid_queries[q].values(), key=lambda x: x['similarity'])['similarity'], reverse=True):
            best_match = None
            for person, match in sorted(valid_queries[q].items(), key=lambda item: item[1]['similarity'], reverse=True):
                if person not in used_persons:
                    best_match = match
                    used_persons.add(person)
                    break
            if best_match is None:
                # If all persons are used, take the best available (duplicate)
                best_match = max(valid_queries[q].values(), key=lambda x: x['similarity'])
            best_match_with_q = best_match.copy()
            best_match_with_q["query_face"] = q
            fallback_assignment.append(best_match_with_q)

        final_map = {m["query_face"]: m for m in fallback_assignment}
        result = [final_map.get(q, None) for q in matrix.keys()]
        return result

    # 5. Brute-force permutation for best average score
    best_score = -1
    best_assignment = None

    for selected_persons_tuple in permutations(possible_persons, len(query_faces)):
        total_score = 0.0
        valid = True
        current_assignment = []

        for query_face, person in zip(query_faces, selected_persons_tuple):
            match = valid_queries.get(query_face, {}).get(person)
            if match is None:
                valid = False
                break
            total_score += match["similarity"]
            match_with_q = match.copy()
            match_with_q["query_face"] = query_face
            current_assignment.append(match_with_q)

        if not valid:
            continue

        average_score = total_score / len(query_faces)
        if average_score > best_score:
            best_score = average_score
            best_assignment = current_assignment

    if best_assignment is None:
        return [None] * len(matrix.keys())

    # 6. Final result mapping
    assigned_map = {m["query_face"]: m for m in best_assignment}
    final_result = [assigned_map.get(q, None) for q in matrix.keys()]
    return final_result

# ============================================================
# 3. MAIN FUNCTION
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python face_search.py <image_path>")
        return

    image_path = sys.argv[1]
    print(f"Processing: {image_path}")

    # ============================================================
    # ૧. InsightFace લોડ કરો
    # ============================================================
    try:
        app = FaceAnalysis(name='buffalo_l', root='insightface_models')
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"ERROR loading InsightFace: {e}")
        return

    # ============================================================
    # ૨. ઇમેજ વાંચો અને ચહેરા શોધો
    # ============================================================
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return

    faces = app.get(img)
    # Left-to-Right sort (if multiple faces)
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f.bbox[0])

    if len(faces) == 0:
        print("No faces detected in the image.")
        print("\nPERSON LEVEL AGGREGATION")
        print("1. PERSON NONE")
        print("Person Score : 0.0000")
        return

    print(f"Detected {len(faces)} face(s)")

    # ============================================================
    # ૩. ક્વેરી એમ્બેડિંગ્સ બનાવો
    # ============================================================
    query_embeddings = []
    for i, face in enumerate(faces):
        embedding = face.embedding / np.linalg.norm(face.embedding)
        query_embeddings.append({
            "query_face": f"face_{i}",
            "embedding": embedding.tolist()
        })

    # ============================================================
    # ૪. JSON ફાઇલમાંથી લેબલ વાંચો
    # ============================================================
    import json
    try:
        with open("fresh_labels.json", "r", encoding="utf-8") as f:
            label_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read fresh_labels.json - {e}")
        return

        db_faces = []
    for idx, item in enumerate(label_data):
        label = item.get("person_label")
        embedding = item.get("embedding")
        filename = item.get("filename", "unknown")  # <--- આ નવું છે
        if label is None or label == "SKIP" or embedding is None:
            continue
        emb_np = np.array(embedding)
        norm = np.linalg.norm(emb_np)
        if norm == 0:
            continue
        emb_np = emb_np / norm
        db_faces.append((idx, label, emb_np, face_finder))  # <--- અહીં filename ઉમેર્યું

    print(f"[OK] Loaded {len(db_faces)} valid labeled faces from JSON")

    if not db_faces:
        print("No valid faces found in fresh_labels.json")
        print("\nPERSON LEVEL AGGREGATION")
        print("1. PERSON NONE")
        print("Person Score : 0.0000")
        return

    # ============================================================
    # ૫. સિમિલેરિટી ગણો
    # ============================================================
        query_face_matches = {}
    for q_data in query_embeddings:
        q_face = q_data["query_face"]
        q_emb = np.array(q_data["embedding"])
        query_face_matches[q_face] = []
        for db_id, label, db_emb, face_finder in db_faces:  # <--- અહીં filename ઉમેર્યું
            similarity = float(np.dot(q_emb, db_emb))
            query_face_matches[q_face].append({
                "person": label,
                "similarity": similarity,
                "face_id": db_id,
                "filename":face_finder  # <--- અહીં filename ઉમેર્યું
            })
        query_face_matches[q_face].sort(key=lambda x: x["similarity"], reverse=True)

    # ============================================================
    # ૬. Global Assignment ચલાવો
    # ============================================================
    persons_list = ['A', 'B', 'C', 'D']
    result = find_best_global_assignment(
        query_embeddings,
        query_face_matches,
        persons_list
    )

    # ============================================================
    # ૭. Decision Logic (STRONG / AMBIGUOUS / WEAK)
    # ============================================================
    if result:
        for match in result:
            if match is None:
                continue
            q_face = match["query_face"]
            all_matches = query_face_matches.get(q_face, [])
            if len(all_matches) < 2:
                match["decision"] = "STRONG"
                continue

            top_score = all_matches[0]["similarity"]
            second_score = all_matches[1]["similarity"]
            margin = top_score - second_score

            # નવો સાચો ક્રમ: પહેલાં Top Score જુઓ
            if top_score > 0.80:
                decision = "STRONG"
            elif top_score > 0.65 and margin > 0.08:
                decision = "GOOD"
            elif margin < 0.05:
                decision = "AMBIGUOUS"
            elif top_score < 0.50:
                decision = "WEAK"
            else:
                decision = "WEAK"

            match["decision"] = decision

       # ============================================================
    # ૮. Final Ranking Print કરો (ફક્ત Strong/Good સ્કોર, Left-to-Right)
    # ============================================================
    if result:
        # 1) ફક્ત 0.30 થી વધુ સ્કોરવાળા મેચ રાખો (Weak દૂર કરો)
        filtered = [
            match for match in result 
            if match is not None and match['similarity'] > 0.30
        ]
        
        # 2) તેમને Face Index (face_0, face_1...) ના ક્રમમાં ગોઠવો (Left-to-Right)
        filtered.sort(key=lambda x: int(x['query_face'].split('_')[1]))
        
        print("\nPERSON LEVEL AGGREGATION")
        for idx, match in enumerate(filtered, start=1):
            print(f"{idx}. PERSON {match['person']}")
            print(f"Person Score : {match['similarity']:.4f}")
            print(f"Decision : {match.get('decision', 'UNKNOWN')}")
    else:
        print("\nPERSON LEVEL AGGREGATION")
        print("1. PERSON NONE")
        print("Person Score : 0.0000")
        print("Decision : NO_MATCH")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
 