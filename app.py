from flask import Flask, request, jsonify
from model import FaceModel
from encryption import EmbeddingEncryption
import os
import sys
import cv2
import base64
import json

app = Flask(__name__)

dataset_folder = "dataset"
db_embeddings = {}
encrypted_db = {} 
face_model = FaceModel(db_embeddings=db_embeddings)

encryptor = EmbeddingEncryption()
encryption_key = encryptor.get_key()

with open('encryption_key.txt', 'w') as f:
    f.write(encryption_key)
print(f"🔐 Encryption key saved to encryption_key.txt", file=sys.stderr, flush=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("\n" + "="*50, file=sys.stderr, flush=True)
print("Starting Face Recognition", file=sys.stderr, flush=True)
print("="*50 + "\n", file=sys.stderr, flush=True)

# Build database
print("📂 Loading dataset...", file=sys.stderr, flush=True)
for filename in os.listdir(dataset_folder):

    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        person_name = os.path.splitext(filename)[0]
        img_path = os.path.join(dataset_folder, filename)

        try:
            # Encrypted
            encrypted_embedding = encryptor.encrypt_embedding(face_model.get_embedding(img_path))
            db_embeddings[person_name] = encrypted_embedding

            print(f"🍀 Added encrypted {person_name} to database", file=sys.stderr, flush=True)
        
        except Exception as e:
            print(f"📢 Failed to add {person_name}: {str(e)}", file=sys.stderr, flush=True)

print(f"\n✅ Database loaded with {len(db_embeddings)} entries\n", file=sys.stderr, flush=True)

@app.route("/detect_face", methods=["POST"])
def detect_face():

    print("\n" + "="*50, file=sys.stderr, flush=True)
    print("📸 /detect_face called", file=sys.stderr, flush=True)
    print("="*50, file=sys.stderr, flush=True)
    
    if "image" not in request.files:
        print("📢 No image file provided", file=sys.stderr, flush=True)
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_path = "temp_detect.jpg"
    file.save(img_path)

    try:

        success, face_img, coords = face_model.detect_face(img_path)
        
        if not success:
            if os.path.exists(img_path):
                os.remove(img_path)
            return jsonify({
                "success": False,
                "error": "No face detected in image"
            }), 400
        
        original_img = cv2.imread(img_path)

        cv2.rectangle(original_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
        
        _, buffer_original = cv2.imencode('.jpg', original_img)
        original_base64 = base64.b64encode(buffer_original).decode('utf-8')
        
        _, buffer_face = cv2.imencode('.jpg', face_img)
        face_base64 = base64.b64encode(buffer_face).decode('utf-8')
        
        if os.path.exists(img_path):
            os.remove(img_path)
        
        print("✅ Face detection successful\n", file=sys.stderr, flush=True)
        
        # Visualization
        accept_header = request.headers.get('Accept', '')
        if 'text/html' in accept_header or request.args.get('visualize') == 'true':
            html_response = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Face Detection Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 1200px;
                        margin: 50px auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .image-section {{
                        display: flex;
                        gap: 20px;
                        justify-content: center;
                        flex-wrap: wrap;
                        margin: 30px 0;
                    }}
                    .image-box {{
                        text-align: center;
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border: 2px solid #ddd;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="image-section">
                        <div class="image-box">
                            <h3>📷 Original Image</h3>
                            <img src="data:image/jpeg;base64,{original_base64}" alt="Original">
                        </div>
                        <div class="image-box">
                            <h3>👤 Detected Face (cropped)</h3>
                            <img src="data:image/jpeg;base64,{face_base64}" alt="Face">
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            return html_response, 200, {'Content-Type': 'text/html'}
        
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
        print(f"📢 Error: {str(e)}", file=sys.stderr, flush=True)
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": str(e)}), 500

@app.route("/verify_face", methods=["POST"])
def verify_face():
    print("\n" + "="*50, file=sys.stderr, flush=True)
    print("🔐 /verify_face called", file=sys.stderr, flush=True)
    print("="*50, file=sys.stderr, flush=True)
    
    if "image" not in request.files:
        print("📢 No image file provided", file=sys.stderr, flush=True)
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_path = "temp_upload.jpg"
    file.save(img_path)

    try:
        # Liveness Detection
        is_live, liveness_score = face_model.check_liveness(img_path)
        
        # Face Matching
        match_found = False
        matched_person = None
        similarity = 0.0
        
        if is_live:
            # Decrypted
            decrypted_db = encryptor.decrypt_database(db_embeddings)
            face_model.db_embeddings = decrypted_db
            
            match_found, matched_person, similarity = face_model.compare_face(img_path)
        else:
            print("\n📢 Skipping face matching - liveness check failed", file=sys.stderr, flush=True)

        if os.path.exists(img_path):
            os.remove(img_path)

        # Visualization
        matched_text = (
            f'<h3>Match with: {matched_person}</h3>'
            if match_found else
            ''
        )
        accept_header = request.headers.get('Accept', '')
        if 'text/html' in accept_header or request.args.get('visualize') == 'true':
            html_response = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Face Detection Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 1200px;
                        margin: 50px auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .text-section {{
                        display: flex;
                        gap: 20px;
                        justify-content: center;
                        flex-wrap: wrap;
                        margin: 30px 0;
                    }}
                    .text-box {{
                        text-align: center;
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="text-section">
                        <div class="text-box">
                            <h3>Liveness: {bool(is_live)}</h3>
                        </div>
                        <div class="text-box">
                            <h3>Recognize: {bool(match_found)}</h3>
                            {matched_text}
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            return html_response, 200, {'Content-Type': 'text/html'}

        result = {
            "is_live": bool(is_live),
            "liveness_score": float(round(liveness_score, 2)),
            "match_found": bool(match_found),
            "matched_person": matched_person,
            "similarity_score": float(round(similarity, 2))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr, flush=True)
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)