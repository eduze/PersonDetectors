import copy
import cv2
import time

from RealtimeCapture import RealtimeCapture


class HogPersonDetector:
    def __init__(self, preview):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detectPersons(self, colour_frame, gray_frame):
        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(colour_frame)

        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
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
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = RealtimeCapture("test_videos/test_office.mp4")
    while True:
        r, frame = cap.read()
        if not r:
            break
        start_time = time.time()
        frame = cv2.resize(frame, (1280, 720))  # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Haar-cascade classifier needs a grayscale image

        rects, weights = hog.detectMultiScale(frame)

        end_time = time.time()

        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print("Frame Time: {}".format(end_time - start_time))
        cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
