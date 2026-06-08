"""Face mesh detection using MediaPipe and region mask generation."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional
import urllib.request
import os


LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
LANDMARKER_PATH = "/tmp/face_landmarker.task"


def _ensure_model():
    if not os.path.exists(LANDMARKER_PATH):
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)


class FaceMeshDetector:
    """MediaPipe face mesh detector."""

    def __init__(self):
        _ensure_model()
        base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]

        h, w = img_bgr.shape[:2]
        landmarks = []
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])

        return np.array(landmarks, dtype=np.int32)

    def __del__(self):
        if hasattr(self, "detector"):
            self.detector.close()


def make_region_masks(landmarks: np.ndarray, img_shape: tuple) -> dict[str, np.ndarray]:
    """
    Create region masks for different facial areas.

    Regions: forehead, cheeks (left/right), nose, chin
    """
    h, w = img_shape[:2]
    masks = {}

    forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    left_cheek_indices = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152,
                          377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                          356, 389, 251, 284, 332, 297, 338]

    right_cheek_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                           152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                           234, 127, 162, 21, 54, 103, 67, 109]

    nose_indices = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12,
                    13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

    chin_indices = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                    356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54,
                    21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 176]

    def create_mask_from_indices(indices, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(indices) == 0:
            return mask
        valid_indices = [i for i in indices if i < len(landmarks)]
        if len(valid_indices) < 3:
            return mask
        points = landmarks[valid_indices]
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    masks["forehead"] = create_mask_from_indices(forehead_indices[:20], (h, w))
    masks["nose"] = create_mask_from_indices(nose_indices, (h, w))
    masks["chin"] = create_mask_from_indices([152, 175, 200, 201, 18, 421, 406, 335, 273], (h, w))

    left_side = landmarks[landmarks[:, 0] < w // 2]
    if len(left_side) > 3:
        hull = cv2.convexHull(left_side)
        masks["left_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["left_cheek"], hull, 255)

    right_side = landmarks[landmarks[:, 0] >= w // 2]
    if len(right_side) > 3:
        hull = cv2.convexHull(right_side)
        masks["right_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["right_cheek"], hull, 255)

    if "left_cheek" in masks and "right_cheek" in masks:
        masks["cheeks"] = cv2.bitwise_or(masks["left_cheek"], masks["right_cheek"])
        del masks["left_cheek"]
        del masks["right_cheek"]

    return masks
