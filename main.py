
import cv2

from src.object_perception import ObjectDetection
from src.lane_detection import LaneDetection
from src.pothole_detection import RoadSurfacePerception


def main():

    # Video input path
    video_path = r"/home/sathya/projects/adas-road-perception/data/driving_video.mp4"

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video")
        return

    # Initialize object detection module
    object_detection = ObjectDetection()
    lane_detection = LaneDetection()
    pothole_detection = RoadSurfacePerception()

    print("ADAS Object Detection Started")

    while True:

        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        # Perform object detection
        annotated_frame = object_detection.object_detection(frame)
        # Perform lane detection
        annotated_frame, _ = lane_detection.detect_lane_lines(annotated_frame)
        # Perform pothole detection
        annotated_frame, _ = pothole_detection.detect_potholes(annotated_frame)
        # Display annotated video
        cv2.imshow(
            "ADAS - Object Detection",
            annotated_frame
        )

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("ADAS Object Perception Completed")


if __name__ == "__main__":
    main()

