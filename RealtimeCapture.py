from time import time

import cv2


class RealtimeCapture:
    def __init__(self, params):
        self.capture = cv2.VideoCapture(params)
        self.video_frame_time = 0
        self.start_time = None

    def get_time(self):
        return self.early_fetch_time

    def read(self):
        if self.start_time is None:
            self.start_time = time()

        process_time = time() - self.start_time

        ret, frame = self.capture.read()
        while self.video_frame_time + process_time > self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000:
            ret, frame = self.capture.read()

        self.start_time = time()
        self.video_frame_time = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

        return ret, frame


if __name__ == "__main__":
    rc = RealtimeCapture(("",))

    start_time = time.time()
    while True:
        r, frame = rc.read()
        current_time = time.time()
        frame_time = rc.get_time()

        if r:
            cv2.putText(frame, "Video Time: " + str(frame_time), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
            cv2.putText(frame, "Wall Time: " + str(current_time - start_time), (30, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 255, 0))
            cv2.imshow("Preview", frame)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break
