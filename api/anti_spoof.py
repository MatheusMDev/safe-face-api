# anti_spoof.py
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

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
def detect_phone_borders(bgr,
                         border_frac=0.12,
                         min_diff_strong=35,  # Reduzido de 35 para 25
                         min_diff_weak=20):   # Reduzido de 18 para 10
    """
    Detecta padrão típico de celular em pé:
      - faixas mais escuras nas bordas esquerda/direita
      - centro mais claro (tela)

    Não depende de um threshold absoluto de "escuridão".
    Usa diferença relativa de brilho entre centro e bordas.
    Retorna score 0..1.
    """
    if bgr is None:
        return 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bw = max(4, int(border_frac * w))
    if w <= 2 * bw:
        return 0.0

    left   = gray[:, :bw]
    right  = gray[:, w - bw:]
    center = gray[:, bw:w - bw]

    m_left   = float(np.mean(left))
    m_right  = float(np.mean(right))
    m_center = float(np.mean(center))

    side_max = max(m_left, m_right)
    diff     = m_center - side_max   # pode ser negativa

    # Se o centro nem é mais claro, não parece tela
    if diff <= 0:
        return 0.0

    # Mapeia diff em [0,1] considerando thresholds fraco/forte
    if diff <= min_diff_weak:
        score = 0.0
    elif diff >= min_diff_strong:
        score = 1.0
    else:
        score = (diff - min_diff_weak) / float(min_diff_strong - min_diff_weak)

    return float(score)

# =============================
# ROI em torno da face para procurar o celular
# =============================
def extract_phone_roi(bgr, face_box, margin_x=0.5, margin_y=0.4):
    """
    Cria um retângulo maior ao redor da face para procurar bordas de celular.
    margin_x / margin_y são frações da largura/altura da face.
    """
    h, w, _ = bgr.shape
    if face_box is None:
        return bgr

    x1, y1, x2, y2 = face_box
    fw = x2 - x1
    fh = y2 - y1

    # expande em torno da face
    cx1 = max(0, int(x1 - fw * margin_x))
    cx2 = min(w, int(x2 + fw * margin_x))
    cy1 = max(0, int(y1 - fh * margin_y))
    cy2 = min(h, int(y2 + fh * margin_y))

    roi = bgr[cy1:cy2, cx1:cx2].copy()
    return roi

# =============================
# HEURÍSTICA GLARE: PONTOS MUITO CLAROS
# =============================
def detect_glare_global(bgr, thr=220, min_pixels=400):  # Reduzido de 235 para 230
    """
    Detecta brilho forte (glare) na imagem toda.
    """
    if bgr is None:
        return 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > thr  # diminui o limiar de brilho para detectar reflexos mais baixos
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
        # Fallback if no face_box is provided, use the whole image as face_crop
        face_crop = bgr

    prob_real_cnn = predict_spoof_prob_real_from_face(face_crop)

    # ROI maior em torno da face para procurar celular
    # Using the adjusted margins as observed in the previous run's context
    phone_roi   = extract_phone_roi(bgr, face_box, margin_x=0.5, margin_y=0.4)
    phone_score = detect_phone_borders(phone_roi, border_frac=0.12, min_diff_strong=35, min_diff_weak=20)

    # glare global ainda na imagem toda
    glare_score = detect_glare_global(bgr, thr=230, min_pixels=500)

    prob_adj = prob_real_cnn

    # =========================
    # Regra dura: celular forte => FAKE direto
    # =========================
    if phone_score >= 0.30:   # Alterado para 0.30
        prob_adj = 0.0
        label = "FAKE"
    else:
        # Penalizações combinando glare + borda de celular
        if phone_score >= 0.20 or glare_score >= 0.18:
            prob_adj = prob_real_cnn * 0.10
        elif phone_score >= 0.08 or glare_score >= 0.10:
            prob_adj = prob_real_cnn * 0.30

        label = "REAL" if prob_adj >= SPOOF_THRESHOLD else "FAKE"

    print(
        f"[HYBRID] cnn={prob_real_cnn:.3f} | phoneBorder={phone_score:.3f} "
        f"| glareGlob={glare_score:.3f} | adj={prob_adj:.3f} "
        f"| thr={SPOOF_THRESHOLD:.3f} -> {label}"
    )

    if debug:
        # debug visual das bordas no ROI do celular
        try:
            gray = cv2.cvtColor(phone_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            bw = max(4, int(0.10 * w))
            dbg = phone_roi.copy()
            cv2.rectangle(dbg, (0, 0), (bw, h), (255, 0, 0), 2)
            cv2.rectangle(dbg, (w - bw, 0), (w, h), (255, 0, 0), 2)
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"phoneBorder={phone_score:.3f}, glare={glare_score:.3f}")
            plt.show()
        except Exception as e:
            print("Erro no debug visual de phone_roi:", e)

    return label, prob_real_cnn, phone_score, glare_score, prob_adj