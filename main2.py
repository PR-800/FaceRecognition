import cv2
import numpy as np
import time
from datetime import datetime
import os

# Import the advanced face matcher
try:
    from face_matcher_deep import get_face_matcher, TORCH_AVAILABLE
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("⚠️ Advanced face recognition not available")
    print("Make sure face_matcher_deep.py is in the same directory")


class CameraFaceRecognition:
    """Real-time face recognition from camera"""
    
    def __init__(self, db_dir="./dataset", threshold=0.6):
        self.db_dir = db_dir
        self.threshold = threshold
        
        # Initialize face matcher
        if ADVANCED_AVAILABLE:
            print("Initializing advanced face recognition system...")
            self.matcher = get_face_matcher(db_dir)
            
            # Build/load database
            print("Loading face database...")
            self.matcher.build_database(force_rebuild=False)
            
            self.use_advanced = TORCH_AVAILABLE
        else:
            self.use_advanced = False
            print("⚠️ Running without advanced features")
        
        # Camera settings
        self.camera = None
        self.frame_skip = 15  # Process every Nth frame for performance
        self.frame_count = 0
        
        # Recognition state
        self.last_recognition = None
        self.last_distance = None
        self.recognition_cooldown = 2.0  # seconds
        self.last_recognition_time = 0
    
    def init_camera(self, camera_id=0):
        """Initialize camera"""
        print(f"Opening camera {camera_id}...")
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return False
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera opened successfully")
        return True
    
    def detect_face_quick(self, frame):
        """Quick face detection for display (no processing)"""
        h, w = frame.shape[:2]
        
        # Use Haar Cascade for quick detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        return faces
    
    def recognize_frame(self, frame):
        """Recognize face in frame using advanced system"""
        if not self.use_advanced:
            return None, None
        
        try:
            # Save temporary image
            temp_path = "temp_camera_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect and process face
            face, quality = self.matcher.detector.process_face(temp_path, verbose=False)
            
            if face is None:
                return None, None
            
            # Extract embedding
            embedding = self.matcher.embedder.extract_embedding(face)
            
            if embedding is None:
                return None, None
            
            # Search in database
            label, distance = self.matcher.database.search(embedding)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return label, distance
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, None
    
    def draw_ui(self, frame, faces):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw face rectangles
        for (x, y, w_face, h_face) in faces:
            # Determine color based on recognition
            if self.last_recognition and self.last_distance is not None:
                if self.last_distance < self.threshold:
                    color = (0, 255, 0)  # Green for match
                    label = self.last_recognition
                else:
                    color = (0, 165, 255)  # Orange for unknown
                    label = "Unknown"
            else:
                color = (255, 255, 255)  # White for detecting
                label = "Detecting..."
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw distance if available
            if self.last_distance is not None:
                dist_text = f"Dist: {self.last_distance:.3f}"
                cv2.putText(frame, dist_text, (x, y + h_face + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw status bar at top
        status_bg = np.zeros((40, w, 3), dtype=np.uint8)
        status_bg[:] = (50, 50, 50)
        
        # System status
        status_text = "Face Recognition Status: ON" if self.use_advanced else "Basic Mode"
        cv2.putText(status_bg, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Database info
        # if self.use_advanced and hasattr(self.matcher, 'database'):
        #     db_text = f"Database: {len(self.matcher.database.labels)} faces"
        #     cv2.putText(status_bg, db_text, (10, 50), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Threshold
        threshold_text = f"Threshold: {self.threshold:.2f}"
        cv2.putText(status_bg, threshold_text, (w - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw instructions at bottom
        instructions_bg = np.zeros((90, w, 3), dtype=np.uint8)
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save snapshot",
            "Press '+/-' to adjust threshold",
            "Press 'r' to Rebuild database"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(instructions_bg, text, (10, 20 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        frame = np.vstack([status_bg, frame, instructions_bg])

        return frame
    
    def run(self):
        """Main camera loop"""
        if not self.init_camera():
            return
        
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Quick face detection for display
                faces = self.detect_face_quick(frame)
                
                # Process recognition every N frames
                current_time = time.time()
                if (self.frame_count % self.frame_skip == 0 and 
                    len(faces) > 0 and 
                    current_time - self.last_recognition_time > self.recognition_cooldown):
                    
                    if self.use_advanced:
                        print("🔍 Recognizing face...")
                        label, distance = self.recognize_frame(frame)
                        
                        if label is not None:
                            self.last_recognition = label
                            self.last_distance = distance
                            self.last_recognition_time = current_time
                            
                            if distance < self.threshold:
                                print(f"✅ Recognized: {label} (distance: {distance:.4f})")
                            else:
                                print(f"❓ Unknown person (closest: {label}, distance: {distance:.4f})")
                
                # Draw UI
                display_frame = self.draw_ui(frame.copy(), faces)
                
                # Show frame
                cv2.imshow('Face Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Snapshot saved: {filename}")
                elif key == ord('+') or key == ord('='):
                    self.threshold += 0.05
                    print(f"Threshold increased to {self.threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.threshold = max(0.1, self.threshold - 0.05)
                    print(f"Threshold decreased to {self.threshold:.2f}")
                elif key == ord('r'):
                    if self.use_advanced:
                        print("Rebuilding database...")
                        self.matcher.build_database(force_rebuild=True)
                        print("✅ Database rebuilt")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            print("✅ Camera closed")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Face Recognition')
    parser.add_argument('--db', default='./dataset', help='Database directory')
    parser.add_argument('--threshold', type=float, default=0.6, 
                       help='Recognition threshold (lower = stricter)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db):
        print(f"⚠️ Database directory not found: {args.db}")
        print("Please create the directory and add face images")
        return
    
    # Count images in database
    image_files = [f for f in os.listdir(args.db) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"⚠️ No images found in {args.db}")
        print("Please add face images to the database directory")
        return
    
    print(f"Found {len(image_files)} images in database")
    
    # Initialize and run
    recognizer = CameraFaceRecognition(db_dir=args.db, threshold=args.threshold)
    recognizer.run()


if __name__ == "__main__":
    main()