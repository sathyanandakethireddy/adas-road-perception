
from asyncio import wait
import cv2
import numpy as np


class LaneDetection:

    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150

    def region_of_interest(self, edges):
        cv2.waitKey(0)
        height, width = edges.shape[:2]

        # Focus on the road region
        roi_vertices = np.array([[
            (0, height),
            (int(width*0.35), int(height*0.55)),
            (int(width*0.65), int(height*0.55)),
            (width, height)
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges

    def detect_lane_lines(self, frame):
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            raise ValueError("frame must be a non-empty image")

        # 1. Convert frame to grayscale
        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Detect edges
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        # 4. Select road region
        roi_edges = self.region_of_interest(edges)

        # 5. Detect line segments
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=100
        )

        # 6. Draw detected line segments
        annotated_frame = frame.copy()
        if lines is not None:
            for line in lines:
                coordinates = line.reshape(-1)
                x1, y1, x2, y2 = map(int, coordinates[:4])
                cv2.line(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

        return annotated_frame, roi_edges