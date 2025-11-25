# anti_spoof.py
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import ANTI_MODEL_PATH, THRESH_PATH, IMG_SIZE, SPOOF_THRESHOLD_DEFAULT

# =============================
# LOAD ANTI-SPOOF MODEL
# =============================
print("Carregando modelo anti-spoof...")

try:
    ANTI_MODEL = tf.keras.models.load_model(ANTI_MODEL_PATH, compile=False)
    print(f"Modelo anti-spoof carregado de: {ANTI_MODEL_PATH}")
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


# =============================
# ANTI-SPOOF CNN (FACE CROP)
# =============================
def preprocess_spoof_pil(pil_img: Image.Image) -> np.ndarray:
    arr = np.asarray(pil_img.resize(IMG_SIZE).convert("RGB"), dtype=np.float32)
    return arr[None, ...]


def predict_spoof_prob_real_from_face(face_bgr: np.ndarray) -> float:
    """
    CNN só no rosto recortado.
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    x = preprocess_spoof_pil(pil)
    prob = float(ANTI_MODEL.predict(x, verbose=0)[0][0])
    return prob


# =============================
# HEURÍSTICA PHONE: BORDAS PRETAS + CENTRO CLARO
# =============================
def detect_phone_borders(
    bgr: np.ndarray,
    border_frac: float = 0.12,
    dark_thr: int = 80,
    center_thr: int = 110,
    min_contrast: float = 20.0,
) -> float:
    """
    Procura padrão típico de celular em pé:
      - faixas escuras nas bordas esquerda/direita
      - centro bem mais claro

    Retorna score 0..1 (quanto maior, mais "cara de celular").
    """
    if bgr is None:
        return 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bw = max(4, int(border_frac * w))  # largura das faixas laterais
    if w <= 2 * bw:
        return 0.0

    left = gray[:, :bw]
    right = gray[:, w - bw :]
    center = gray[:, bw : w - bw]

    m_left = float(np.mean(left))
    m_right = float(np.mean(right))
    m_center = float(np.mean(center))

    # pelo menos UMA borda bem escura
    side_dark = (m_left < dark_thr) or (m_right < dark_thr)
    # centro razoavelmente iluminado
    center_bright = m_center > center_thr

    # contraste centro versus borda mais clara (a melhor das duas)
    side_max = max(m_left, m_right)
    contrast = max(0.0, m_center - side_max)

    if side_dark and center_bright and contrast >= min_contrast:
        score = contrast / 255.0  # normaliza contraste para 0..1
    else:
        score = 0.0

    return score


# =============================
# HEURÍSTICA GLARE: PONTOS MUITO CLAROS
# =============================
def detect_glare_global(
    bgr: np.ndarray, thr: int = 245, min_pixels: int = 400
) -> float:
    """
    Mede % de pixels muito claros (glare geral).
    Não derruba REAL sozinho, só em conjunto com bordas.
    """
    if bgr is None:
        return 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > thr
    count = int(np.sum(mask))
    if count < min_pixels:
        return 0.0
    return count / float(gray.size)


# =============================
# HÍBRIDO SIMPLES
# =============================
def classify_spoof_hybrid(
    bgr: np.ndarray,
    face_box: tuple | None,
    debug: bool = False,
):
    """
    - Recebe frame completo (bgr) + bounding box do rosto (face_box)
    - CNN roda só no rosto recortado
    - Heurísticas de borda preta + glare atuam sobre a imagem inteira
    """
    # crop do rosto
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        face_crop = bgr[y1:y2, x1:x2].copy()
    else:
        face_crop = bgr

    prob_real_cnn = predict_spoof_prob_real_from_face(face_crop)
    phone_score = detect_phone_borders(bgr)
    glare_score = detect_glare_global(bgr)

    prob_adj = prob_real_cnn

    # Regra:
    # - Só derruba para FAKE quando parecer muito celular:
    #   bordas fortes OU bordas moderadas + glare.
    if phone_score >= 0.10:
        prob_adj = prob_real_cnn * 0.20  # derruba bem
    elif phone_score >= 0.05 and glare_score >= 0.04:
        prob_adj = prob_real_cnn * 0.25

    # NÃO mexe em prob_adj só por glare; evita matar REAL por causa de luz.
    label = "REAL" if prob_adj >= SPOOF_THRESHOLD else "FAKE"

    if debug:
        print(
            f"[HYBRID] cnn={prob_real_cnn:.3f} | phoneBorder={phone_score:.3f} "
            f"| glareGlob={glare_score:.3f} | adj={prob_adj:.3f} "
            f"| thr={SPOOF_THRESHOLD:.3f} -> {label}"
        )

    return label, prob_real_cnn, phone_score, glare_score, prob_adj