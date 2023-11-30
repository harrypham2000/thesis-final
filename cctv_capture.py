import cv2
import os
from datetime import datetime

class CCTV_Capture:
    def __init__(self, rtsp_url, save_path):
        self.rtsp_url = rtsp_url
        self.save_path = save_path
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            timestampt_string=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestampt_string, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(self.save_path, 'frame.jpg'), frame)
        else:
            print("Cannot capture the frame")

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    rtsp_url = "rtsp://giahung:1309800ok@171.240.152.2:13280/Streaming/Channels/201"
    save_path = "C:/Users/Harry/Desktop/thesis2023/thesis/source"
    cctv = CCTV_Capture(rtsp_url, save_path)
    cctv.capture_frame()
    cctv.release()

