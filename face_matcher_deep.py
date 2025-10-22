import cv2
import numpy as np
import os.path
from scipy.spatial.distance import cosine
import urllib.request

class AccurateFaceMatcher:
    """
    Face matcher using OpenCV DNN for both face detection and feature extraction.
    """
    def __init__(self):
        # Load OpenCV DNN face detector
        print("Loading face detector...")
        self.detector_modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
        self.detector_configFile = "deploy.prototxt"
        self._download_model_files()

        try:
            self.detector_net = cv2.dnn.readNetFromCaffe(self.detector_configFile, self.detector_modelFile)
            print("✅ Face detector loaded (DNN)")
        except cv2.error:
            print("⚠️ Failed to load DNN face detector.")
            self.detector_net = None

        # Load pre-trained face recognition model
        print("Loading face recognition model...")
        self.recognizer_modelFile = "face-recognition-sface_2021dec.onnx"
        self._download_recognizer_model()
        
        try:
            self.recognizer_net = cv2.dnn.readNet(self.recognizer_modelFile)
            print("✅ Face recognizer loaded (SFACE)")
        except cv2.error:
            print("⚠️ Failed to load face recognizer.")
            self.recognizer_net = None

    def _download_model_files(self):
        """Download face detection models if they don't exist"""
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
        caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

        if not os.path.exists(self.detector_configFile):
            print(f"Downloading {self.detector_configFile}...")
            urllib.request.urlretrieve(base_url + "deploy.prototxt", self.detector_configFile)
            print("✅ Downloaded deploy.prototxt")

        if not os.path.exists(self.detector_modelFile):
            print(f"Downloading {self.detector_modelFile}...")
            urllib.request.urlretrieve(caffemodel_url, self.detector_modelFile)
            print("✅ Downloaded res10_300x300_ssd_iter_140000.caffemodel")
            
    def _download_recognizer_model(self):
        """Download face recognizer model if it doesn't exist"""
        recognizer_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        if not os.path.exists(self.recognizer_modelFile):
            print(f"Downloading {self.recognizer_modelFile}...")
            try:
                urllib.request.urlretrieve(recognizer_url, self.recognizer_modelFile)
                print(f"✅ Downloaded {self.recognizer_modelFile}")
            except Exception as e:
                print(f"❌ Failed to download {self.recognizer_modelFile}: {e}")
                # Provide a helpful message if download fails
                print("Please try downloading the file manually from the URL above and place it in the same directory as your script.")

    def detect_and_crop_face(self, image_path):
        """
        Detect and crop the face with the highest confidence from an image.
        Returns the cropped face or None if no face is found.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return None

        if self.detector_net is None:
            return None

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.detector_net.setInput(blob)
        detections = self.detector_net.forward()

        max_confidence = 0.5  # Set a minimum confidence threshold
        best_box = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_confidence:
                max_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype("int")

        if best_box is not None:
            (startX, startY, endX, endY) = best_box
            face = img[startY:endY, startX:endX]
            return face
        
        return None

    def extract_features(self, face_image):
        """
        Extract a 128-dimensional feature vector from a face image.
        """
        if face_image is None or self.recognizer_net is None:
            return None
        
        # The model expects a 112x112 image
        face_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        self.recognizer_net.setInput(face_blob)
        features = self.recognizer_net.forward()
        return features.flatten()

    def match_faces(self, db_dir, unknown_image_path):
        """
        Match an unknown face with a database of known faces.
        """
        unknown_face = self.detect_and_crop_face(unknown_image_path)

        if unknown_face is None:
            print("No face detected in the unknown image.")
            return f"{unknown_image_path},no_persons_found\n"

        unknown_features = self.extract_features(unknown_face)

        if unknown_features is None:
            print("Failed to extract features from the unknown face.")
            return f"{unknown_image_path},no_persons_found\n"

        best_match = "unknown_person"
        best_similarity = -1.0 
        threshold = 0.5  # Adjust this threshold based on your needs

        db_files = [f for f in os.listdir(db_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for filename in db_files:
            db_path = os.path.join(db_dir, filename)
            db_face = self.detect_and_crop_face(db_path)

            if db_face is None:
                continue

            db_features = self.extract_features(db_face)

            if db_features is None:
                continue

            # Using cosine similarity to compare features
            similarity = 1 - cosine(unknown_features, db_features)

            if similarity > best_similarity:
                best_similarity = similarity
                if similarity > threshold:
                    best_match = os.path.splitext(filename)[0]

        if best_match != "unknown_person":
            print(f"✅ Match found: {best_match} (Similarity: {best_similarity:.4f})")
            return f"{unknown_image_path},{best_match}\n"
        else:
            print(f"❌ No match found. Best similarity: {best_similarity:.4f}")
            return f"{unknown_image_path},unknown_person\n"

# Global instance
_face_matcher = None

def get_face_matcher_deep():
    """Get or create face matcher instance"""
    global _face_matcher
    if _face_matcher is None:
        _face_matcher = AccurateFaceMatcher()
    return _face_matcher

# Example of how to use the new class
# if __name__ == '__main__':
#     matcher = AccurateFaceMatcher()
    
    # Create a 'database' directory with images of known people
    # Create a 'test_images' directory with images to test against the database
    
    # Example usage:
    # result = matcher.match_faces('database', 'test_images/some_person.jpg')
    # print(result)