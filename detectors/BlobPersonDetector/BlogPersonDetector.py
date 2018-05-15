import copy
import cv2


class BlobPersonDetector:
    def __init__(self, preview):
        self.frame_original = None
        self.firstFrame = None
        self.countour_treshold = 5000

    def detectPersons(self, colour_frame, gray_frame):
        if gray_frame is None:
            gray_frame = cv2.cvtColor(colour_frame, cv2.COLOR_BGR2GRAY)

        frame = gray_frame

        # resize the frame to 800xM, convert it to grayscale, and blur it
        # frame = imutils.resize(frame, width=width)

        frame = colour_frame
        self.frame_original = copy.copy(frame)
        gray_clear = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray_clear, (21, 21), 0)

        # if the first frame is None, initialize it
        if self.firstFrame is None:
            self.firstFrame = gray
            return []

            # compute the absolute difference between the current frame and
            # first frame
        frameDelta = cv2.absdiff(self.firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours

        detections = []

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.countour_treshold:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)

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
