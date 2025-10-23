# dlib: pip install https://github.com/omwaman1/dlib-for-python3.13.2/releases/download/dlib/dlib-19.24.99-cp313-cp313-win_amd64.whl

import cv2
import numpy as np
import os
import json
from pathlib import Path
import urllib.request
import traceback

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available, install with: pip install torch torchvision")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available, install with: pip install faiss-cpu")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("⚠️ dlib not available (optional for alignment), install with: pip install dlib")


class ImageQualityAssessor:
    """Assess image quality for face recognition"""
    
    @staticmethod
    def detect_blur(image):
        """Detect blur using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    @staticmethod
    def check_brightness(image):
        """Check image brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return brightness
    
    @staticmethod
    def estimate_noise(image):
        """Estimate image noise level"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use standard deviation as noise estimate
        noise_level = np.std(gray)
        return noise_level
    
    @staticmethod
    def assess_quality(image, verbose=True):
        """Comprehensive quality assessment"""
        blur_score = ImageQualityAssessor.detect_blur(image)
        brightness = ImageQualityAssessor.check_brightness(image)
        noise = ImageQualityAssessor.estimate_noise(image)
        
        # Quality thresholds
        is_blurry = blur_score < 100  # Lower = more blurry
        is_too_dark = brightness < 50
        is_too_bright = brightness > 200
        is_noisy = noise > 50
        
        quality_report = {
            'blur_score': blur_score,
            'brightness': brightness,
            'noise_level': noise,
            'is_acceptable': not (is_blurry or is_too_dark or is_too_bright or is_noisy),
            'issues': []
        }
        
        if is_blurry:
            quality_report['issues'].append('Image is blurry')
        if is_too_dark:
            quality_report['issues'].append('Image is too dark')
        if is_too_bright:
            quality_report['issues'].append('Image is too bright')
        if is_noisy:
            quality_report['issues'].append('Image is noisy')
        
        if verbose:
            print(f"  📊 Quality Assessment:")
            print(f"     Blur Score: {blur_score:.2f} {'✅' if not is_blurry else '⚠️ Too blurry'}")
            print(f"     Brightness: {brightness:.2f} {'✅' if not (is_too_dark or is_too_bright) else '⚠️ Poor lighting'}")
            print(f"     Noise Level: {noise:.2f} {'✅' if not is_noisy else '⚠️ Too noisy'}")
            print(f"     Overall: {'✅ Good' if quality_report['is_acceptable'] else '⚠️ Poor quality'}")
        
        return quality_report


class FaceDetectorAligner:
    """Advanced face detection with alignment"""
    
    def __init__(self):
        print("Loading face detector and aligner...")
        
        # Load DNN face detector
        self.modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "deploy.prototxt"
        
        if not os.path.exists(self.modelFile) or not os.path.exists(self.configFile):
            self._download_models()
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
            print("✅ DNN Face detector loaded")
        except:
            print("❌ Failed to load DNN detector")
            self.net = None
        
        # Load Haar Cascade as fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load dlib predictor if available
        self.predictor = None
        if DLIB_AVAILABLE:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
                self.dlib_detector = dlib.get_frontal_face_detector()
                print("✅ dlib landmark predictor loaded")
            else:
                print("⚠️ dlib predictor not found (optional)")
    
    def _download_models(self):
        """Download face detection models"""
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
    
    def detect_face(self, image):
        """Detect face and return bounding box"""
        h, w = image.shape[:2]
        
        # Try DNN first
        if self.net is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    (104.0, 177.0, 123.0)
                )
                
                self.net.setInput(blob)
                detections = self.net.forward()
                
                max_confidence = 0
                best_box = None
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > 0.5 and confidence > max_confidence:
                        max_confidence = confidence
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        best_box = box.astype("int")
                
                if best_box is not None:
                    return best_box, max_confidence
            except Exception as e:
                print(f"⚠️ DNN detection failed: {e}")
        
        # Fallback to Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) > 0:
            (x, y, w_face, h_face) = max(faces, key=lambda f: f[2] * f[3])
            return np.array([x, y, x + w_face, y + h_face]), 0.8
        
        return None, 0
    
    def get_landmarks(self, image, bbox):
        """Get facial landmarks"""
        if not DLIB_AVAILABLE or self.predictor is None:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape = self.predictor(gray, dlib_rect)
            
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            return landmarks
        except:
            return None
    
    def align_face(self, image, landmarks):
        """Align face based on eye positions"""
        if landmarks is None or len(landmarks) < 68:
            return image
        
        # Get eye centers
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get center between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                      (left_eye[1] + right_eye[1]) / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply transformation
        h, w = image.shape[:2]
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def process_face(self, image_path, verbose=True):
        """Complete face processing pipeline"""
        if verbose:
            print(f"\n🔍 Processing: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return None, None
        
        # Assess quality
        quality = ImageQualityAssessor.assess_quality(image, verbose)
        
        # Detect face
        bbox, confidence = self.detect_face(image)
        if bbox is None:
            print("❌ No face detected")
            return None, quality
        
        if verbose:
            print(f"  ✅ Face detected (confidence: {confidence:.2f})")
        
        # Get landmarks
        landmarks = self.get_landmarks(image, bbox)
        
        if landmarks is not None and verbose:
            print(f"  ✅ {len(landmarks)} facial landmarks detected")
            
            # Show landmark visualization
            vis_image = image.copy()
            for (x, y) in landmarks:
                cv2.circle(vis_image, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            # Save visualization
            output_path = image_path.replace('.', '_landmarks.')
            cv2.imwrite(output_path, vis_image)
            if verbose:
                print(f"  💾 Landmarks saved to: {output_path}")
        
        # Align face
        aligned = self.align_face(image, landmarks) if landmarks is not None else image
        
        # Crop face with margin
        x1, y1, x2, y2 = bbox
        margin = int(0.2 * (x2 - x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        face = aligned[y1:y2, x1:x2]
        
        return face, quality


class ResNetEmbedder:
    """ResNet-based face embedding generator"""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ResNet embeddings")
        
        print("Loading ResNet-50 model...")
        
        # Load pre-trained ResNet-50
        self.model = models.resnet50(pretrained=True)
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"✅ ResNet-50 loaded on {self.device}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_embedding(self, face_image):
        """Extract 2048-d embedding from face"""
        if face_image is None:
            return None
        
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                embedding = self.model(input_tensor)
            
            # Flatten and normalize
            embedding = embedding.squeeze().cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            traceback.print_exc()
            return None


class FAISSDatabase:
    """FAISS vector database for face embeddings"""
    
    def __init__(self, dimension=2048):
        if not FAISS_AVAILABLE:
            print("⚠️ FAISS not available, using numpy fallback")
            self.use_faiss = False
        else:
            self.use_faiss = True
            self.index = faiss.IndexFlatL2(dimension)
            print(f"✅ FAISS index created (dimension: {dimension})")
        
        self.embeddings = []
        self.labels = []
        self.dimension = dimension
    
    def add_embedding(self, embedding, label):
        """Add embedding to database"""
        if embedding is None:
            return
        
        if self.use_faiss:
            embedding_reshaped = embedding.reshape(1, -1).astype('float32')
            self.index.add(embedding_reshaped)
        
        self.embeddings.append(embedding)
        self.labels.append(label)
    
    def search(self, query_embedding, k=1):
        """Search for nearest neighbors"""
        if query_embedding is None or len(self.embeddings) == 0:
            return None, float('inf')
        
        query = query_embedding.reshape(1, -1).astype('float32')
        
        if self.use_faiss:
            distances, indices = self.index.search(query, k)
            best_idx = indices[0][0]
            best_distance = distances[0][0]
        else:
            # Numpy fallback
            distances = [np.linalg.norm(query - emb) for emb in self.embeddings]
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
        
        return self.labels[best_idx], best_distance
    
    def save(self, path):
        """Save database to disk"""
        data = {
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'labels': self.labels
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"✅ Database saved to {path}")
    
    def load(self, path):
        """Load database from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.embeddings = [np.array(emb) for emb in data['embeddings']]
        self.labels = data['labels']
        
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.dimension)
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.index.add(embeddings_array)
        
        print(f"✅ Database loaded from {path} ({len(self.labels)} faces)")


class AdvancedFaceMatcher:
    """Complete face recognition system"""
    
    def __init__(self, db_dir="/dataset"):
        self.db_dir = db_dir
        self.detector = FaceDetectorAligner()
        
        if TORCH_AVAILABLE:
            self.embedder = ResNetEmbedder()
            self.database = FAISSDatabase()
            self.use_advanced = True
            print("✅ Using advanced ResNet + FAISS system")
        else:
            self.use_advanced = False
            print("⚠️ Using basic system (install torch and faiss for better accuracy)")
        
        self.db_path = "face_database.json"
    
    def build_database(self, force_rebuild=False):
        """Build embedding database from images"""
        if not self.use_advanced:
            print("⚠️ Advanced features not available")
            return
        
        if not force_rebuild and os.path.exists(self.db_path):
            print(f"Loading existing database from {self.db_path}")
            self.database.load(self.db_path)
            return
        
        # Clear existing database when rebuilding (Press 'r')
        if force_rebuild:
            print("🗑️ Clearing existing database...")
            self.database = FAISSDatabase() 
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                print("✅ Old database deleted")
        
        print(f"\n{'='*60}")
        print("Building Face Database")
        print(f"{'='*60}")
        
        if not os.path.exists(self.db_dir):
            print(f"❌ Database directory not found: {self.db_dir}")
            return
        
        image_files = [f for f in os.listdir(self.db_dir) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"❌ No images found in {self.db_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        for filename in image_files:
            image_path = os.path.join(self.db_dir, filename)
            label = filename.rsplit('.', 1)[0]
            
            # Process face
            face, quality = self.detector.process_face(image_path, verbose=True)
            
            if face is None:
                print(f"⚠️ Skipping {filename}")
                continue
            
            # Extract embedding
            print(f"  🧠 Extracting embedding...")
            embedding = self.embedder.extract_embedding(face)
            
            if embedding is not None:
                self.database.add_embedding(embedding, label)
                print(f"  ✅ Added {label} to database")
            else:
                print(f"  ❌ Failed to extract embedding")
        
        # Save database
        self.database.save(self.db_path)
        
        print(f"\n{'='*60}")
        print(f"✅ Database built: {len(self.database.labels)} faces")
        print(f"{'='*60}\n")
    
    def match_face(self, image_path, threshold=0.6):
        """Match face from image"""
        if not self.use_advanced:
            return f"{image_path},no_persons_found\n"
        
        print(f"\n{'='*60}")
        print(f"Matching: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Process face
        face, quality = self.detector.process_face(image_path, verbose=True)
        
        if face is None:
            print("❌ No face detected")
            return f"{image_path},no_persons_found\n"
        
        # Extract embedding
        print(f"  🧠 Extracting embedding...")
        embedding = self.embedder.extract_embedding(face)
        
        if embedding is None:
            print("❌ Failed to extract embedding")
            return f"{image_path},no_persons_found\n"
        
        # Search in database
        print(f"  🔍 Searching in database...")
        label, distance = self.database.search(embedding)
        
        print(f"\n  Best match: {label}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Threshold: {threshold}")
        
        if distance < threshold:
            print(f"\n✅ MATCH: {label}")
            result = f"{image_path},{label}\n"
        else:
            print(f"\n❌ NO MATCH (unknown person)")
            result = f"{image_path},unknown_person\n"
        
        print(f"{'='*60}\n")
        return result


# Main interface functions
_face_matcher = None

def get_face_matcher(db_dir="/dataset"):
    """Get or create face matcher instance"""
    global _face_matcher
    if _face_matcher is None:
        _face_matcher = AdvancedFaceMatcher(db_dir)
    return _face_matcher


# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = get_face_matcher("/dataset")
    
    # Build database (run once or when adding new faces)
    matcher.build_database(force_rebuild=True)
    
    # Match a face
    result = matcher.match_face("unknown_face.jpg", threshold=0.6)
    print(result)