import tkinter as tk
import util
import cv2
from PIL import Image, ImageTk
import os
import requests
import datetime
import subprocess

from face_matcher import get_face_matcher 
from face_matcher_deep import get_face_matcher_deep

class App:
    def __init__(self):
        # Main window
        self.main_window = tk.Tk()
        self.main_window.geometry("900x430")
        self.main_window.title("Face Recognition App")

        # Verify button
        self.verify_button = util.get_button(self.main_window, "Verify", "green", self.verify)
        self.verify_button.place(x=660, y=300)

        # Liveness check button
        # self.liveness_button = util.get_button(self.main_window, "Liveness Check", "grey", self.check_liveness)
        # self.liveness_button.place(x=660, y=360)

        # Webcam 
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label .place(x=30, y=10, width=600, height=400)

        # Database directory
        self.db_dir = "./dataset"
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        # Log file
        self.log_path = "./access_log.txt"

        # # API endpoint
        # self.api_url = "http://localhost:5000"

        self.add_webcam(self.webcam_label)

    def add_webcam(self, label):
        if "cap" not in self.__dict__:
            self.cap = cv2.VideoCapture(1) 

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        
        self.most_recent_capture_pil = Image.fromarray(img)
        imgTk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)

        self._label.imagetk = imgTk
        self._label.configure(image=imgTk)
        
        self._label.after(20, self.process_webcam)

    # def update_status(self, message, color="black"):
    #     """Update status label"""
    #     self.status_label.config(text=message, fg=color)
    #     self.main_window.update()

    # def check_liveness(self):
        # """Check if face is real or fake"""
        # self.update_status("🔍 Checking liveness...", "blue")
        
        # try:
        #     # Encode current frame
        #     img_b64 = self.image_to_base64(self.most_recent_capture_arr)
            
        #     # Call API
        #     response = requests.post(
        #         f"{self.api_url}/api/liveness-check",
        #         json={'image': img_b64},
        #         timeout=10
        #     )
            
        #     result = response.json()
            
        #     if result.get('success'):
        #         if result.get('is_real'):
        #             self.update_status(
        #                 f"✅ Real Face (Confidence: {result['confidence']:.2f})",
        #                 "green"
        #             )
        #         else:
        #             self.update_status(
        #                 f"❌ Fake Face Detected! (Confidence: {result['confidence']:.2f})",
        #                 "red"
        #             )
                
        #         self.display_result(result)
        #     else:
        #         self.update_status(f"❌ Error: {result.get('error')}", "red")
                
        # except requests.exceptions.ConnectionError:
        #     self.update_status("❌ Cannot connect to API", "red")
        # except Exception as e:
        #     self.update_status(f"❌ Error: {str(e)}", "red")
    
    def verify(self):
        unknown_image_path = "./.tmp.jpg"
        cv2.imwrite(unknown_image_path, self.most_recent_capture_arr)
        
        # output = subprocess.check_output(["face_recognition", self.db_dir, unknown_image_path])
        # matcher = get_face_matcher()
        matcher = get_face_matcher_deep()
        output = matcher.match_faces(self.db_dir, unknown_image_path)

        name = output.split(',')[1][:-1]  # Remove newline
        print("Identified as:", name)
            
        if name == "no_persons_found":
            util.msg_box("No face detected", "Please show your face clearly")
        elif name == "unknown_person":
            util.msg_box("Access Denied", "Face not recognized")
        else:
            util.msg_box("Success", f"Welcome, {name}")
            with open(self.log_path, 'a') as f:
                f.write(f"{name}, {datetime.datetime.now()}\n")
                f.close()
        
        os.remove(unknown_image_path)

    def register(self):
        pass
    
    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()