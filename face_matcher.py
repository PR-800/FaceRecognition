import cv2
import numpy as np
import os.path
from scipy.spatial.distance import cosine

class SimpleFaceMatcher:
    """
    Face matcher using OpenCV DNN (no mediapipe needed!)
    """
    def __init__(self):
        # Load OpenCV DNN face detector
        print("Loading face detector...")
        
        # Download model files if not exist
        self.modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "deploy.prototxt"
        
        # Initialize Haar Cascade as backup (ตอนนี้ initialize เสมอ)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Check if model files exist, if not download them
        if not os.path.exists(self.modelFile) or not os.path.exists(self.configFile):
            print("Downloading face detection model...")
            self._download_models()
        
        # Load the DNN model
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
            print("✅ Face detector loaded (DNN)")
        except:
            print("⚠️ DNN failed, using Haar Cascade only")
            self.net = None
    
    def _download_models(self):
        """Download face detection models"""
        import urllib.request
        
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
        
        files = {
            "deploy.prototxt": base_url + "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": 
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"✅ Downloaded {filename}")
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
    
    def detect_and_crop_face(self, image_path):
        """
        Detect and crop face from image
        Returns None if no face found
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        h, w = img.shape[:2]
        face = None
        
        # Try DNN detector first
        if self.net is not None:
            try:
                # Prepare image for DNN
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(img, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    (104.0, 177.0, 123.0)
                )
                
                self.net.setInput(blob)
                detections = self.net.forward()
                
                # Find face with highest confidence
                max_confidence = 0
                best_box = None
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > 0.5 and confidence > max_confidence:
                        max_confidence = confidence
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        best_box = box.astype("int")
                
                if best_box is not None:
                    (x, y, x2, y2) = best_box
                    # Add margin
                    margin = int(0.2 * (x2 - x))
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    
                    face = img[y:y2, x:x2]
                    return face
            except Exception as e:
                print(f"⚠️ DNN detection failed: {e}")
        
        # Fallback to Haar Cascade
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                print("⚠️ No face detected")
                return None
            
            # Get largest face
            (x, y, w_face, h_face) = max(faces, key=lambda f: f[2] * f[3])
            
            # Add margin
            margin = int(0.2 * w_face)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w_face = min(w - x, w_face + 2 * margin)
            h_face = min(h - y, h_face + 2 * margin)
            
            face = img[y:y+h_face, x:x+w_face]
            return face
        except Exception as e:
            print(f"❌ Face detection failed: {e}")
            return None
    
    def extract_features(self, face_image):
        """Extract simple features from face"""
        if face_image is None:
            return None
        
        try:
            # Resize to standard size
            face = cv2.resize(face_image, (128, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram as features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            return hist
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return None
    
    def match_faces(self, db_dir, unknown_image_path):
        """
        Match unknown face with database
        Returns string similar to face_recognition output
        """
        try:
            # Get face from unknown image
            unknown_face = self.detect_and_crop_face(unknown_image_path)
            
            # ตรงนี้สำคัญ - ถ้าไม่เจอหน้า return ทันที
            if unknown_face is None:
                print("No face detected in unknown image")
                return f"{unknown_image_path},no_persons_found\n"
            
            # Extract features
            unknown_features = self.extract_features(unknown_face)
            
            if unknown_features is None:
                print("Failed to extract features")
                return f"{unknown_image_path},no_persons_found\n"
            
            # Compare with all images in database
            best_match = None
            best_distance = float('inf')
            threshold = 1  # Lower = more strict
            
            db_files = [f for f in os.listdir(db_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(db_files) == 0:
                print("No registered faces in database")
                return f"{unknown_image_path},unknown_person\n"
            
            for filename in db_files:
                db_path = os.path.join(db_dir, filename)
                db_face = self.detect_and_crop_face(db_path)
                
                if db_face is None:
                    continue
                
                db_features = self.extract_features(db_face)
                
                if db_features is None:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(unknown_features - db_features)
                print(f"Distance to {filename}: {distance:.4f}")
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = filename.rsplit('.', 1)[0]  # Remove extension
            
            # Format output like face_recognition
            if best_match and best_distance < threshold:
                output = f"{unknown_image_path},{best_match}\n"
                print(f"✅ Match found: {best_match} (distance: {best_distance:.4f})")
            else:
                output = f"{unknown_image_path},unknown_person\n"
                print(f"❌ No match (best distance: {best_distance:.4f})")
            
            return output
            
        except Exception as e:
            print(f"❌ Error in match_faces: {e}")
            import traceback
            traceback.print_exc()
            return f"{unknown_image_path},no_persons_found\n"

# Global instance
_face_matcher = None

def get_face_matcher():
    """Get or create face matcher instance"""
    global _face_matcher
    if _face_matcher is None:
        _face_matcher = SimpleFaceMatcher()
    return _face_matcher