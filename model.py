from deepface import DeepFace
import cv2
import numpy as np
from utils import load_image

class FaceModel:

    def __init__(self, db_embeddings={}):
        self.db_embeddings = db_embeddings

    # ------------------------------
    # Liveness
    # ------------------------------
    def check_liveness(self, img_path, threshold=0.5):
        img = load_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur Detection (Laplacian Variance Method)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness Check
        brightness = np.mean(gray)

        score = (lap_var / 200.0 + brightness / 255.0) / 2.0
        score = min(max(score, 0.0), 1.0)

        is_live = score > threshold 
        return is_live, score

    # ------------------------------
    # Matching
    # ------------------------------
    def get_embedding(self, img_path):
        obj = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=True)
        embedding = obj[0]["embedding"]
        return np.array(embedding)

    def compare_face(self, img_path, threshold=0.5):
        query_embedding = self.get_embedding(img_path)

        best_match = None
        best_score = 0.0

        for person, db_embedding in self.db_embeddings.items():
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            if similarity > best_score:
                best_score = similarity
                best_match = person

        match_found = best_score >= threshold
        return match_found, best_match, best_score
