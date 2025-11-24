# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ANTI_MODEL_PATH = os.path.join(BASE_DIR, "anti_spoof.keras")
THRESH_PATH     = os.path.join(BASE_DIR, "threshold.json")

IMG_SIZE = (160, 160)  # mesmo do treino

# threshold de similaridade para reconhecimento facial
FACE_THRESHOLD = 0.60

# anti-spoof fallback
SPOOF_THRESHOLD_DEFAULT = 0.51

# Firebase
FIREBASE_CREDENTIALS = os.path.join(BASE_DIR, "serviceAccountKey.json")
FIREBASE_FACES_COLLECTION = "faces"  # nome da collection no Firestore