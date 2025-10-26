from flask import Flask, request, jsonify
from model import FaceModel
import os
import sys
import cv2
import base64

app = Flask(__name__)

dataset_folder = "dataset"
db_embeddings = {}
face_model = FaceModel(db_embeddings=db_embeddings)

# Force stdout/stderr to flush immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("\n" + "="*50, file=sys.stderr, flush=True)
print("🚀 Starting Face Recognition API", file=sys.stderr, flush=True)
print("="*50 + "\n", file=sys.stderr, flush=True)

# Build database from dataset
print("📁 Loading dataset...", file=sys.stderr, flush=True)
for filename in os.listdir(dataset_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        person_name = os.path.splitext(filename)[0]
        img_path = os.path.join(dataset_folder, filename)
        try:
            db_embeddings[person_name] = face_model.get_embedding(img_path)
            print(f"✓ Added {person_name} to database", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"✗ Failed to add {person_name}: {str(e)}", file=sys.stderr, flush=True)

print(f"\n✅ Database loaded with {len(db_embeddings)} entries\n", file=sys.stderr, flush=True)

@app.route("/detect_face", methods=["POST"])
def detect_face():
    """
    GET endpoint to test face detection
    Returns: detected face coordinates and cropped face as base64
    """
    print("\n" + "="*50, file=sys.stderr, flush=True)
    print("📸 /detect_face endpoint called", file=sys.stderr, flush=True)
    print("="*50, file=sys.stderr, flush=True)
    
    if "image" not in request.files:
        print("❌ No image file provided", file=sys.stderr, flush=True)
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_path = "temp_detect.jpg"
    file.save(img_path)
    print(f"💾 Image saved to {img_path}", file=sys.stderr, flush=True)

    try:
        # Detect face
        success, face_img, coords = face_model.detect_face(img_path)
        
        if not success:
            if os.path.exists(img_path):
                os.remove(img_path)
            return jsonify({
                "success": False,
                "error": "No face detected in image"
            }), 400
        
        # Convert cropped face to base64 for visualization
        _, buffer = cv2.imencode('.jpg', face_img)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        if os.path.exists(img_path):
            os.remove(img_path)
        
        print("✅ Face detection successful\n", file=sys.stderr, flush=True)
        
        return jsonify({
            "success": True,
            "face_coordinates": {
                "x1": int(coords[0]),
                "y1": int(coords[1]),
                "x2": int(coords[2]),
                "y2": int(coords[3])
            },
            "cropped_face_base64": face_base64,
            "message": "Face detected successfully"
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr, flush=True)
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": str(e)}), 500

@app.route("/verify_face", methods=["POST"])
def verify_face():
    print("\n" + "="*50, file=sys.stderr, flush=True)
    print("🔐 /verify_face endpoint called", file=sys.stderr, flush=True)
    print("="*50, file=sys.stderr, flush=True)
    
    if "image" not in request.files:
        print("❌ No image file provided", file=sys.stderr, flush=True)
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_path = "temp_upload.jpg"
    file.save(img_path)
    print(f"💾 Image saved to {img_path}", file=sys.stderr, flush=True)

    try:
        # Step 1: Liveness Detection
        print("\n🔍 Step 1: Liveness Detection", file=sys.stderr, flush=True)
        is_live, liveness_score = face_model.check_liveness(img_path)
        
        # Step 2: Face Matching (only if live)
        match_found = False
        matched_person = None
        similarity = 0.0
        
        if is_live:
            print("\n🔍 Step 2: Face Matching", file=sys.stderr, flush=True)
            match_found, matched_person, similarity = face_model.compare_face(img_path)
        else:
            print("\n⚠️  Skipping face matching - liveness check failed", file=sys.stderr, flush=True)

        if os.path.exists(img_path):
            os.remove(img_path)

        result = {
            "is_live": bool(is_live),
            "liveness_score": float(round(liveness_score, 2)),
            "match_found": bool(match_found),
            "matched_person": matched_person,
            "similarity_score": float(round(similarity, 2))
        }
        
        print("\n📋 Final Result:", file=sys.stderr, flush=True)
        print(f"   Is Live: {result['is_live']}", file=sys.stderr, flush=True)
        print(f"   Liveness Score: {result['liveness_score']}", file=sys.stderr, flush=True)
        print(f"   Match Found: {result['match_found']}", file=sys.stderr, flush=True)
        print(f"   Matched Person: {result['matched_person']}", file=sys.stderr, flush=True)
        print(f"   Similarity: {result['similarity_score']}", file=sys.stderr, flush=True)
        print("="*50 + "\n", file=sys.stderr, flush=True)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr, flush=True)
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🌐 Starting Flask server on http://0.0.0.0:5000", file=sys.stderr, flush=True)
    print("📡 Endpoints available:", file=sys.stderr, flush=True)
    print("   POST /detect_face  - Test face detection", file=sys.stderr, flush=True)
    print("   POST /verify_face  - Full verification (liveness + matching)", file=sys.stderr, flush=True)
    print("\n" + "="*50 + "\n", file=sys.stderr, flush=True)
    app.run(host="0.0.0.0", port=5000, debug=True)