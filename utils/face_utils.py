import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# =========================================================
# SETTINGS
# =========================================================

EMBEDDINGS_FILE = Path(
    "data/face_embeddings.pkl"
)


# =========================================================
# FACE MODEL
# =========================================================

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# =========================================================
# CREATE EMBEDDING
# =========================================================

def create_embedding(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    faces = face_app.get(
        image
    )

    if len(faces) == 0:
        return None

    # Use the largest detected face
    face = max(
        faces,
        key=lambda x: (
            x.bbox[2] - x.bbox[0]
        ) * (
            x.bbox[3] - x.bbox[1]
        )
    )

    return face.embedding


# =========================================================
# SAVE EMBEDDING
# =========================================================

def save_embedding(
    student_id,
    embedding
):

    EMBEDDINGS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if EMBEDDINGS_FILE.exists():

        try:

            with open(
                EMBEDDINGS_FILE,
                "rb"
            ) as file:

                embeddings = pickle.load(
                    file
                )

        except Exception:

            embeddings = {}

    else:

        embeddings = {}


    embeddings[str(student_id)] = (
        embedding
    )


    with open(
        EMBEDDINGS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            embeddings,
            file
        )


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

def load_embeddings():

    if not EMBEDDINGS_FILE.exists():

        return {}


    try:

        with open(
            EMBEDDINGS_FILE,
            "rb"
        ) as file:

            return pickle.load(
                file
            )

    except Exception:

        return {}


# =========================================================
# FIND STUDENT
# =========================================================

def find_student(
    face_embedding,
    threshold=0.45
):

    embeddings = load_embeddings()

    if not embeddings:

        return None


    current = np.asarray(
        face_embedding,
        dtype=np.float32
    )

    current = current / (
        np.linalg.norm(current)
        + 1e-8
    )


    best_student = None
    best_score = -1


    for student_id, saved in (
        embeddings.items()
    ):

        saved = np.asarray(
            saved,
            dtype=np.float32
        )

        saved = saved / (
            np.linalg.norm(saved)
            + 1e-8
        )


        score = float(
            np.dot(
                current,
                saved
            )
        )


        if score > best_score:

            best_score = score

            best_student = student_id


    if best_score >= threshold:

        return best_student


    return None


# =========================================================
# BUILD EMBEDDINGS FROM REGISTERED STUDENTS
# =========================================================

def build_embeddings_from_students(
    students
):

    if students is None:
        return {}


    if students.empty:
        return {}


    embeddings = {}


    for _, student in students.iterrows():

        student_id = str(
            student["student_id"]
        )

        photo_path = student[
            "photo_path"
        ]


        if not photo_path:
            continue


        photo_path = Path(
            str(photo_path)
        )


        if not photo_path.exists():
            continue


        embedding = create_embedding(
            photo_path
        )


        if embedding is None:
            continue


        embeddings[
            student_id
        ] = embedding


    if embeddings:

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


    return embeddings
