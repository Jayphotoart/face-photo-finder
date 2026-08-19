import psycopg2
import numpy as np


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()


cursor.execute("""
    SELECT
        fe.id,
        p.filename,
        fe.face_index,
        fe.embedding,
        fl.person_label
    FROM face_embeddings fe
    JOIN photos p
        ON fe.photo_id = p.id
    LEFT JOIN face_labels fl
        ON fe.id = fl.embedding_id
    ORDER BY fe.id;
""")


rows = cursor.fetchall()


print()
print("================================")
print("EMBEDDING QUALITY CHECK")
print("================================")
print()


problems = []


for embedding_id, filename, face_index, embedding_bytes, label in rows:

    embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32
    )


    norm = np.linalg.norm(embedding)

    has_nan = np.isnan(embedding).any()

    has_inf = np.isinf(embedding).any()


    print(
        f"ID {embedding_id:<3} | "
        f"{filename:<22} | "
        f"Face {face_index + 1} | "
        f"Label {str(label):<2} | "
        f"Shape {str(embedding.shape):<8} | "
        f"Norm {norm:.6f} | "
        f"Min {embedding.min():.4f} | "
        f"Max {embedding.max():.4f}"
    )


    if embedding.shape != (512,):

        problems.append(
            (embedding_id, "Wrong dimension")
        )


    if has_nan:

        problems.append(
            (embedding_id, "NaN detected")
        )


    if has_inf:

        problems.append(
            (embedding_id, "Inf detected")
        )


    if norm == 0:

        problems.append(
            (embedding_id, "Zero norm")
        )


print()
print("================================")
print("QUALITY SUMMARY")
print("================================")


print(
    f"Total embeddings: {len(rows)}"
)


print(
    f"Problems found:   {len(problems)}"
)


if problems:

    print()
    print("PROBLEMS:")

    for embedding_id, reason in problems:

        print(
            f"ID {embedding_id}: {reason}"
        )


print()
print("================================")
print("CHECK COMPLETE")
print("================================")


cursor.close()
conn.close()