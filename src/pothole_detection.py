import cv2
from ultralytics import YOLO
class RoadSurfacePerception:
    def __init__(self,pothole_model_path="yolo26n_pothole.pt"):
        # Load pretrained YOLO26 nano model for pothole detection
        self.pothole_model = YOLO(pothole_model_path)

    def detect_potholes(self,frame):
        # Perform pothole detection
        results = self.pothole_model.predict(frame)
        annotated_frame = results[0].plot()
        return annotated_frame, results