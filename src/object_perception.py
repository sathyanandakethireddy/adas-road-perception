
import cv2
from ultralytics import YOLO


class ObjectPerception:

    def __init__(
        self,
        track_model_path="yolo26n.pt",
        estimation_model_path="yolo26n-depth.pt",
    ):

        # Load pretrained YOLO26 nano model
        self.track_model = YOLO(track_model_path)
        self.estimation_model = YOLO(estimation_model_path)

        # COCO class IDs for road objects
        self.classes = [0, 1, 2, 3, 5, 7, 9]

    def track_objects(self, frame):

        # Perform object tracking
        results = self.track_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.classes,
            conf=0.60,
            iou=0.5,
            verbose=False,
        )

        # Get detection results
        result = results[0]

        # Annotate frame with bounding boxes and labels
        annotated_frame = result.plot()

        return annotated_frame, result

    def distance_estimation(self, frame):
        # Perform depth estimation
        results = self.estimation_model(frame)
        return results


class ObjectDetection(ObjectPerception):

    def object_detection(self, frame):
        annotated_frame, track_results = self.track_objects(frame)
        estimation_results = self.distance_estimation(frame)
        depth_map = estimation_results[0].depth.data
        print("Depth Map Values:", depth_map[1,5])
        for box in track_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = depth_map[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            depth = roi.mean()
            label = track_results.names[int(box.cls[0])]
            text = f"{label}: {depth:.2f} m"
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return annotated_frame

