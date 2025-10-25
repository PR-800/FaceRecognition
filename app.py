from flask import Flask, request, jsonify
from model import FaceModel
import os

app = Flask(__name__)

dataset_folder = "dataset"
db_embeddings = {}
face_model = FaceModel(db_embeddings=db_embeddings)

# Build database from dataset
for filename in os.listdir(dataset_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        person_name = os.path.splitext(filename)[0]
        img_path = os.path.join(dataset_folder, filename)
        db_embeddings[person_name] = face_model.get_embedding(img_path)
        print(f"Added {person_name} to database")

@app.route("/verify_face", methods=["POST"])
def verify_face():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_path = os.path.join("temp_upload.jpg")
    file.save(img_path)

    # Liveness
    is_live, liveness_score = face_model.check_liveness(img_path)

    # Matching
    match_found, matched_person, similarity = face_model.compare_face(img_path)

    if os.path.exists(img_path):
        os.remove(img_path)

    return jsonify({
        "is_live": bool(is_live),
        "liveness_score": float(round(liveness_score, 2)),
        "match_found": bool(match_found),
        "matched_person": matched_person,
        "similarity_score": float(round(similarity, 2))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
