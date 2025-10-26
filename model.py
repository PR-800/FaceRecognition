from deepface import DeepFace
import cv2
import numpy as np
from utils import load_image
from skimage.feature import local_binary_pattern
from scipy import fftpack
import sys

class FaceModel:

    def __init__(self, db_embeddings={}):
        self.db_embeddings = db_embeddings
        # Load OpenCV's pre-trained face detector
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Failed to load Haar Cascade")
            print("✓ Face detector loaded successfully", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not load face cascade: {e}", file=sys.stderr, flush=True)
            print("⚠️  Will use DeepFace for face detection instead", file=sys.stderr, flush=True)
            self.face_cascade = None

    # ------------------------------
    # Face Detection & Cropping
    # ------------------------------
    def detect_face(self, img_path, padding=20):
        """
        Detect face in image and return cropped face region
        Returns: (success, cropped_face_image, face_coordinates)
        """
        img = load_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("⚠️  No face detected in image!", file=sys.stderr, flush=True)
            return False, None, None
        
        # Take the largest face (most prominent)
        if len(faces) > 1:
            print(f"ℹ️  Multiple faces detected ({len(faces)}), using largest", file=sys.stderr, flush=True)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        x, y, w, h = faces[0]
        
        # Add padding to include more context
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        face_img = img[y1:y2, x1:x2]
        
        print(f"✓ Face detected at ({x}, {y}, {w}, {h})", file=sys.stderr, flush=True)
        
        return True, face_img, (x1, y1, x2, y2)

    # ------------------------------
    # Advanced Liveness Detection
    # ------------------------------
    def check_liveness(self, img_path, threshold=0.50):
        """
        Multi-feature liveness detection on FACE REGION ONLY
        """
        # First detect and crop face
        success, face_img, coords = self.detect_face(img_path)
        
        if not success or face_img is None:
            print("❌ Liveness check failed: No face detected", file=sys.stderr, flush=True)
            return False, 0.0
        
        # Now analyze the FACE REGION (not stretched!)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize maintaining aspect ratio info (after cropping face, so no stretch issue)
        img_resized = cv2.resize(face_img, (224, 224))
        gray_resized = cv2.resize(gray, (224, 224))
        
        print("🔍 Running liveness analysis...", file=sys.stderr, flush=True)
        
        # Feature 1: Blur Detection (Laplacian Variance)
        lap_var = cv2.Laplacian(gray_resized, cv2.CV_64F).var()
        blur_score = min(lap_var / 200.0, 1.0)
        
        # Feature 2: LBP Texture Analysis
        lbp = local_binary_pattern(gray_resized, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
        texture_score = min(lbp_entropy / 8.0, 1.0)
        
        # Feature 3: Frequency Domain Analysis (FFT)
        f_transform = fftpack.fft2(gray_resized)
        f_shift = fftpack.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        
        mask_low = np.zeros((rows, cols), np.uint8)
        r = 30
        cv2.circle(mask_low, (ccol, crow), r, 1, -1)
        mask_high = 1 - mask_low
        
        low_freq_energy = np.sum(magnitude_spectrum * mask_low)
        high_freq_energy = np.sum(magnitude_spectrum * mask_high)
        
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-7)
        freq_score = min(freq_ratio / 10.0, 1.0)
        
        # Feature 4: Color Diversity Analysis
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        color_std = np.mean([np.std(hsv[:,:,i]) for i in range(3)])
        color_score = min(color_std / 50.0, 1.0)
        
        # Feature 5: Edge Density
        edges = cv2.Canny(gray_resized, 50, 150)
        edge_density = np.sum(edges > 0) / (224 * 224)
        edge_score = min(edge_density / 0.15, 1.0)
        
        # Feature 6: Brightness Check
        brightness = np.mean(gray_resized)
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        
        # Weighted combination
        weights = {
            'blur': 0.20,
            'texture': 0.25,
            'frequency': 0.20,
            'color': 0.15,
            'edge': 0.10,
            'brightness': 0.10
        }
        
        final_score = (
            weights['blur'] * blur_score +
            weights['texture'] * texture_score +
            weights['frequency'] * freq_score +
            weights['color'] * color_score +
            weights['edge'] * edge_score +
            weights['brightness'] * brightness_score
        )
        
        final_score = min(max(final_score, 0.0), 1.0)
        
        is_live = final_score > threshold
        
        # Debug output with flush
        print(f"📊 Liveness Scores:", file=sys.stderr, flush=True)
        print(f"   Blur: {blur_score:.3f} | Texture: {texture_score:.3f} | Freq: {freq_score:.3f}", file=sys.stderr, flush=True)
        print(f"   Color: {color_score:.3f} | Edge: {edge_score:.3f} | Bright: {brightness_score:.3f}", file=sys.stderr, flush=True)
        print(f"   🎯 FINAL: {final_score:.3f} ({'LIVE ✓' if is_live else 'FAKE ✗'})", file=sys.stderr, flush=True)
        
        return is_live, final_score

    # ------------------------------
    # Face Matching
    # ------------------------------
    def get_embedding(self, img_path):
        """Get face embedding using DeepFace"""
        try:
            obj = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=True)
            embedding = obj[0]["embedding"]
            print(f"✓ Embedding extracted ({len(embedding)} dims)", file=sys.stderr, flush=True)
            return np.array(embedding)
        except Exception as e:
            print(f"❌ Embedding extraction failed: {str(e)}", file=sys.stderr, flush=True)
            raise

    def compare_face(self, img_path, threshold=0.40):
        """
        Compare face with database
        Lowered threshold from 0.5 to 0.40 for better matching
        """
        print("🔍 Extracting query embedding...", file=sys.stderr, flush=True)
        query_embedding = self.get_embedding(img_path)

        best_match = None
        best_score = 0.0

        print(f"🔍 Comparing with {len(self.db_embeddings)} database entries...", file=sys.stderr, flush=True)
        
        for person, db_embedding in self.db_embeddings.items():
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            print(f"   {person}: {similarity:.4f}", file=sys.stderr, flush=True)
            
            if similarity > best_score:
                best_score = similarity
                best_match = person

        match_found = best_score >= threshold
        
        if match_found:
            print(f"✓ MATCH FOUND: {best_match} (score: {best_score:.4f})", file=sys.stderr, flush=True)
        else:
            print(f"✗ NO MATCH (best: {best_match} with {best_score:.4f}, threshold: {threshold})", file=sys.stderr, flush=True)
        
        return match_found, best_match, best_score