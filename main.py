import os
from model import FaceModel

dataset_folder = "dataset"
db_embeddings = {}
face_model = FaceModel(db_embeddings=db_embeddings)

# Build database from all images in dataset
for filename in os.listdir(dataset_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        person_name = os.path.splitext(filename)[0]
        img_path = os.path.join(dataset_folder, filename)
        db_embeddings[person_name] = face_model.get_embedding(img_path)
        print(f"Added {person_name} to database")

# Test all images
for filename in os.listdir(dataset_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        test_img_path = os.path.join(dataset_folder, filename)
        print(f"\nTesting {filename}:")

        is_live, liveness_score = face_model.check_liveness(test_img_path)
        print(f"  Liveness: {is_live}, Score: {liveness_score:.2f}")

        match_found, matched_person, similarity = face_model.compare_face(test_img_path)
        print(f"  Match found: {match_found}, Matched person: {matched_person}, Similarity: {similarity:.2f}")
