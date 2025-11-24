# face_utils.py
import numpy as np
import cv2
import insightface
from typing import Tuple, Any

from config import IMG_SIZE

print("Configurando InsightFace...")

MODEL_ROOT = "./insightface_model"

try:
    APP = insightface.app.FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
    APP.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace usando GPU (ctx_id=0).")
except Exception as e:
    print("Falha GPU:", e)
    print("Tentando CPU (ctx_id=-1)...")
    APP = insightface.app.FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
    APP.prepare(ctx_id=-1, det_size=(640, 640))
    print("InsightFace CPU configurado.")


def get_best_face_embedding(bgr: np.ndarray) -> Tuple[np.ndarray | None, Any | None]:
    faces = APP.get(bgr)
    if not faces:
        return None, None

    main_face = max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
    emb = main_face.embedding.astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb, main_face


def extract_face_crop(bgr: np.ndarray, face) -> np.ndarray:
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = bgr[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, IMG_SIZE)
    return face_crop

