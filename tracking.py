import cv2
import numpy as np
from yolo_detect import YOLO
from deepsort import DeepSort

class Tracking:
    def __init__(self):
        self.yolo = YOLO()
        self.deepsort = DeepSort()

    def track_objects(self, frame):
        # Detect objects in the frame with YOLO
        image = Image.fromarray(frame[...,::-1])  # convert BGR image to RGB
        boxs, scores, classes = self.yolo.detect_image(image)
        boxs = np.array(boxs)

        # Convert box coordinates from (top, left, bottom, right) to (x, y, w, h)
        boxs_xywh = []
        for box in boxs:
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            boxs_xywh.append([x, y, w, h])
        boxs_xywh = np.array(boxs_xywh)

        # Track objects with DeepSORT
        outputs = self.deepsort.update(boxs_xywh, scores, frame)

        # Draw bounding boxes and labels on the frame
        for output in outputs:
            x1, y1, x2, y2, identity = output
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(identity), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame