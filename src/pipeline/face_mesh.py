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
        """
        Detect face landmarks.

        Args:
            img_bgr: Input BGR image

        Returns:
            Landmarks as (N, 2) array of (x, y) pixel coordinates, or None
        """
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
