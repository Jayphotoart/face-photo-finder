import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


def get_faces(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                fe.id AS embedding_id,
                p.filename,
                fe.face_index
            FROM face_embeddings fe
            JOIN photos p
                ON p.id = fe.photo_id
            ORDER BY fe.id;
        """)

        return cur.fetchall()


def save_label(conn, embedding_id, label):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO face_labels (embedding_id, person_label)
            VALUES (%s, %s)
            ON CONFLICT (embedding_id)
            DO UPDATE SET person_label = EXCLUDED.person_label;
        """, (embedding_id, label))

    conn.commit()


def main():
    print("=" * 50)
    print("FACE LABELING")
    print("=" * 50)

    conn = psycopg2.connect(**DB_CONFIG)

    faces = get_faces(conn)

    print(f"\nTotal embeddings: {len(faces)}")
    print("Labels: A / B / C / D / SKIP")
    print("Q = Quit\n")

    for i, (embedding_id, filename, face_index) in enumerate(faces, start=1):

        print("-" * 50)
        print(f"Progress       : {i}/{len(faces)}")
        print(f"Embedding ID   : {embedding_id}")
        print(f"Photo          : {filename}")
        print(f"Face           : {face_index}")

        while True:
            label = input(
                "Person label (A/B/C/D/SKIP): "
            ).strip().upper()

            if label in {"A", "B", "C", "D", "SKIP"}:
                break

            if label == "Q":
                print("\nLabeling stopped.")
                conn.close()
                return

            print("Invalid input. Use A, B, C, D, SKIP or Q.")

        save_label(conn, embedding_id, label)

        print(f"Saved: Embedding {embedding_id} -> {label}")

    conn.close()

    print("\n" + "=" * 50)
    print("LABELING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()