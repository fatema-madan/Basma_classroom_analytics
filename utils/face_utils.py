import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


EMBEDDINGS_FILE = Path("data/face_embeddings.pkl")


face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


def create_embedding(image_path):
    """
    Create a face embedding from a student photo.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    faces = face_app.get(image)

    if len(faces) == 0:
        return None

    face = faces[0]

    return face.embedding


def save_embedding(student_id, embedding):
    """
    Save a student's face embedding.
    """

    if EMBEDDINGS_FILE.exists():

        with open(
            EMBEDDINGS_FILE,
            "rb"
        ) as file:

            embeddings = pickle.load(file)

    else:

        embeddings = {}

    embeddings[str(student_id)] = embedding

    EMBEDDINGS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        EMBEDDINGS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            embeddings,
            file
        )


def load_embeddings():
    """
    Load all saved face embeddings.
    """

    if not EMBEDDINGS_FILE.exists():
        return {}

    with open(
        EMBEDDINGS_FILE,
        "rb"
    ) as file:

        return pickle.load(file)


def find_student(face_embedding, threshold=0.45):
    """
    Find the closest registered student.
    """

    embeddings = load_embeddings()

    if not embeddings:
        return None

    best_student = None
    best_score = -1

    face_embedding = np.array(
        face_embedding
    )

    face_embedding = face_embedding / (
        np.linalg.norm(face_embedding) + 1e-8
    )

    for student_id, saved_embedding in embeddings.items():

        saved_embedding = np.array(
            saved_embedding
        )

        saved_embedding = saved_embedding / (
            np.linalg.norm(saved_embedding) + 1e-8
        )

        score = np.dot(
            face_embedding,
            saved_embedding
        )

        if score > best_score:

            best_score = score
            best_student = student_id

    if best_score >= threshold:
        return best_student

    return None
