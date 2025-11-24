# anti_spoof.py
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import ANTI_MODEL_PATH, THRESH_PATH, IMG_SIZE, SPOOF_THRESHOLD_DEFAULT

print("Carregando modelo anti-spoof...")
try:
    ANTI_MODEL = tf.keras.models.load_model(ANTI_MODEL_PATH, compile=False)
    print("Modelo anti-spoof carregado.")
except Exception as e:
    raise RuntimeError(f"Erro ao carregar ANTI_MODEL: {e}")

try:
    with open(THRESH_PATH, "r") as f:
        THRESH_DATA = json.load(f)
    SPOOF_THRESHOLD = float(THRESH_DATA.get("threshold", SPOOF_THRESHOLD_DEFAULT))
    print(f"SPOOF_THRESHOLD carregado: {SPOOF_THRESHOLD:.3f}")
except Exception as e:
    SPOOF_THRESHOLD = SPOOF_THRESHOLD_DEFAULT
    print("⚠️ Usando SPOOF_THRESHOLD padrão:", SPOOF_THRESHOLD)
    print("Erro ao abrir threshold.json:", e)


def preprocess_spoof_pil(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.resize(IMG_SIZE).convert("RGB"), dtype=np.float32)
    return arr[None, ...]


def predict_spoof_prob_real(bgr: np.ndarray) -> float:
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x = preprocess_spoof_pil(pil)
    prob_real = float(ANTI_MODEL.predict(x, verbose=0)[0][0])
    return prob_real

# Heurística 1: procura retângulos grandes com aspecto de tela de celular/monitor.
def detect_phone_like_rectangle(bgr: np.ndarray, min_area_ratio: float = 0.05) -> float:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    img_area = float(h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.02 or area > img_area * 0.95:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4:
            continue

        x, y, cw, ch = cv2.boundingRect(approx)
        ar = max(cw, ch) / max(1, min(cw, ch))

        if 1.2 <= ar <= 2.8:
            score = area / img_area
            best_score = max(best_score, score)

    return best_score

#Heurística 2: detecta manchas muito brilhantes (reflexo de tela).
def detect_screen_glare(bgr: np.ndarray, thr: int = 235, min_pixels: int = 400) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > thr
    count = int(np.sum(mask))
    if count < min_pixels:
        return 0.0
    h, w = gray.shape
    total = h * w
    score = count / float(total)
    return score

#Função principal de decisão anti-spoof.
def classify_spoof_hybrid(
    bgr: np.ndarray,
    spoof_thr: float | None = None,
    phone_weight: float = 0.8,
    phone_min_area: float = 0.01,
    glare_min: float = 0.01,
):
    if spoof_thr is None:
        spoof_thr = SPOOF_THRESHOLD

    prob_real_cnn = predict_spoof_prob_real(bgr)
    phone_score = detect_phone_like_rectangle(bgr, min_area_ratio=phone_min_area)
    glare_score = detect_screen_glare(bgr, thr=235, min_pixels=400)

    prob_real_adj = prob_real_cnn
    if phone_score >= phone_min_area or glare_score >= glare_min:
        prob_real_adj = prob_real_cnn * (1.0 - phone_weight)

    label = "REAL" if prob_real_adj >= spoof_thr else "FAKE"

    return label, prob_real_cnn, phone_score, glare_score, prob_real_adj