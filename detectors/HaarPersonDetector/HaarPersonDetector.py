import copy
import cv2
import os

import time

from RealtimeCapture import RealtimeCapture


class HaarPersonDetector:
    def __init__(self, preview):
        self.person_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_fullbody.xml'))

    def detectPersons(self, colour_frame, gray_frame):
        # detect people in the image
        if gray_frame is None:
            gray_frame = cv2.cvtColor(colour_frame, cv2.COLOR_RGB2GRAY)
        rects = self.person_cascade.detectMultiScale(gray_frame)

        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            detection = PersonDetection()
            detection.person_bound = (x, y, x + w, y + h)
            detection.central_point = (int(x + w / 2), int(y + h / 2))
            detections.append(detection)

        return detections


class PersonDetection:
    '''
    Detection of a person
    '''

    def __init__(self):
        self.tracked_points = {}  # Points detected by OP
        self.person_bound = None  # Boundary of person
        self.central_bound = None  # Boundary of central body of person (no hands and feet for X coordinate)
        self.upper_body_bound = None  # Boundary of upper body of person
        self.central_point = None  # Central point of person
        self.leg_point = None  # Average Feet point of person
        self.leg_count = None  # Number of detected feet
        self.estimated_leg_point = None  # Estimated feet point of person
        self.neck_hip_ankle_ratio = None
        self.neck_hip_knee_ratio = None
        self.head_direction = None
        self.head_direction_error = None
        self.roi = None


if __name__ == "__main__":
    person_cascade = cv2.CascadeClassifier(
        os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_fullbody.xml'))
    cap = RealtimeCapture("test_videos/test_office.mp4")
    while True:
        r, frame = cap.read()
        if not r:
            break

        start_time = time.time()
        frame = cv2.resize(frame, (1280, 720))  # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Haar-cascade classifier needs a grayscale image

        rects = person_cascade.detectMultiScale(gray_frame)

        end_time = time.time()
        print("Elapsed Time:", end_time - start_time)

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("preview", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
