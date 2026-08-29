import pickle
from pathlib import Path

import face_recognition


EMBEDDINGS_FILE = Path("data/face_embeddings.pkl")


def create_embedding(image_path):

    image = face_recognition.load_image_file(
        image_path
    )

    faces = face_recognition.face_encodings(
        image
    )

    if len(faces) == 0:
        return None

    return faces[0]


def save_embedding(student_id, embedding):

    if EMBEDDINGS_FILE.exists():

        with open(
            EMBEDDINGS_FILE,
            "rb"
        ) as file:

            embeddings = pickle.load(file)

    else:

        embeddings = {}

    embeddings[str(student_id)] = embedding

    with open(
        EMBEDDINGS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            embeddings,
            file
        )


def load_embeddings():

    if not EMBEDDINGS_FILE.exists():
        return {}

    with open(
        EMBEDDINGS_FILE,
        "rb"
    ) as file:

        embeddings = pickle.load(file)

    return embeddings