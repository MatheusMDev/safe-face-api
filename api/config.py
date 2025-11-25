# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelo anti-spoof (treinado no Colab)
ANTI_MODEL_PATH = os.path.join(BASE_DIR, "anti_spoof.keras")
THRESH_PATH     = os.path.join(BASE_DIR, "threshold.json")

# MESMO DO TREINO (160x160)
IMG_SIZE = (160, 160)

# Threshold de similaridade para reconhecimento facial (InsightFace)
FACE_THRESHOLD = 0.60

# Fallback se não conseguir ler o threshold.json
SPOOF_THRESHOLD_DEFAULT = 0.51

# Firebase
FIREBASE_CREDENTIALS      = os.path.join(BASE_DIR, "serviceAccountKey.json")
FIREBASE_FACES_COLLECTION = "faces"  # nome da collection no Firestore