from detectors.MaskDetector.MaskRCN.MaskRCNDetector import MaskRCNDetector
from detectors.MaskDetector.MaskRCN.MaskRCNDetector import class_names
from detectors.MaskDetector.MaskRCN.visualize import apply_mask


class MaskPersonDetector:
    def __init__(self, preview=False, target_class_index=1):
        self.preview = preview
        self.mask_rcn = MaskRCNDetector(preview)
        self.target_class_index = target_class_index
        self.last_results = None

    def detectPersons(self, colour, gray):
        self.last_results = self.mask_rcn.process_frame(colour)
        rois, masks, class_ids, scores = self.last_results

        detections = []
        for i, (y1, x1, y2, x2) in enumerate(rois):
            if class_ids[i] != self.target_class_index:
                continue

            if scores[i] < 0.5:
                continue

            detection = PersonDetection()
            detection.person_bound = (x1, y1, x2, y2)
            detection.central_point = (int(x1 + x2), int(y1 + y2))
            detections.append(detection)

        return detections

    def draw_patches(self, frame):
        if frame is None:
            return

        rois, masks, class_ids, scores = self.last_results

        for i in range(len(class_ids)):
            if class_ids[i] == self.target_class_index:
                mask = masks[:, :, i]

                if mask is None:
                    return

                frame = apply_mask(frame, mask, (0, 1, 1))
        return frame


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
