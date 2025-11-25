from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from config import FACE_THRESHOLD
from firebase_client import verify_id_token, save_face_embedding, get_face_embedding
from anti_spoof import classify_spoof_hybrid
from face_utils import get_best_face_embedding
from typing import List

# ============================================
# App Config & OpenAPI (Swagger)
# ============================================
app = FastAPI(
    title="Face Recognition API",
    version="1.0",
    description="API para autenticação/consulta de faces (MVP).",
    openapi_tags=[
        {"name": "Status", "description": "Rotas de monitoramento e saúde da API."},
        {"name": "Reconhecimento Facial", "description": "Operações de cadastro, verificação e listagem de faces."},
    ],
)

# CORS (ajusta depois para a origem real do app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # em produção, restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Models (requests / responses)
# ===========================================

class RegisterFaceRequest(BaseModel):
    """
    Cadastro de face com múltiplas imagens.
    - idToken: token do Firebase
    - images_base64: lista de 2+ imagens
    """
    idToken: str
    images_base64: List[str]



class VerifyFaceRequest(BaseModel):
    """
    Verificação de face:
    - idToken: token de sessão do Firebase (para descobrir uid)
    - image_base64: foto atual para autenticação
    """
    idToken: str
    image_base64: str


# Opcional: model de resposta mais estruturada
class VerifyFaceResponse(BaseModel):
    status: int
    msg: str
    spoof_label: str
    prob_real_cnn: float
    phone_score: float
    glare_score: float
    prob_real_adj: float
    similarity: float
    passed: bool


# ============================================
# Helpers
# ============================================
def decode_base64_to_bgr(image_base64: str) -> np.ndarray:
    """
    Converte uma string base64 em imagem BGR (OpenCV).
    Aceita data URL no formato "data:image/jpeg;base64,...".
    """
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        img_bytes = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail={"status": 400, "msg": "Base64 inválido."})

    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail={"status": 400, "msg": "Não foi possível decodificar a imagem."})
    return bgr


# ============================================
# Healthcheck
# ============================================
@app.get(
    "/health",
    tags=["Status"],
    summary="Verificar saúde da API",
    responses={
        200: {
            "description": "OK",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "msg": "API On-line."
                    }
                }
            },
        },
        500: {
            "description": "Erro interno",
            "content": {
                "application/json": {
                    "examples": {
                        "falha_simulada": {
                            "summary": "Exemplo de falha simulada",
                            "value": {"status": 500, "msg": "Erro interno: Falha simulada"}
                        },
                    }
                }
            },
        },
    },
)
async def health():
    try:
        # raise Exception("Falha simulada")
        return {"status": 200, "msg": "API On-line"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": 500, "msg": f"Erro interno: {str(e)}"}
        )


# ============================================
# Reconhecimento Facial - Cadastro (ENROLL)
# ============================================
@app.post(
    "/register-face",
    tags=["Reconhecimento Facial"],
    summary="Cadastrar face para um cliente (uid do Firebase)",
    responses={
        201: {
            "description": "Face cadastrada",
            "content": {
                "application/json": {
                    "example": {
                        "status": 201,
                        "msg": "Face cadastrada com sucesso."
                    }
                }
            },
        },
        400: {
            "description": "Requisição inválida / Anti-spoof reprovou",
            "content": {
                "application/json": {
                    "example": {
                        "status": 400,
                        "msg": "Anti-spoof reprovou a imagem."
                    }
                }
            },
        },
        401: {
            "description": "idToken inválido",
            "content": {
                "application/json": {
                    "example": {
                        "status": 401,
                        "msg": "idToken inválido ou expirado."
                    }
                }
            },
        },
        500: {
            "description": "Erro interno",
            "content": {
                "application/json": {
                    "example": {
                        "status": 500,
                        "msg": "Erro interno ao cadastrar face."
                    }
                }
            },
        },
    },
)
async def register_face(body: RegisterFaceRequest):
    try:
        # -------------------------------------------
        # 1) Validar idToken
        # -------------------------------------------
        try:
            uid = verify_id_token(body.idToken)
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail={"status": 401, "msg": f"idToken inválido: {str(e)}"}
            )

        # -------------------------------------------
        # 2) Checar quantidade mínima = 2 imagens
        # -------------------------------------------
        if not body.images_base64 or len(body.images_base64) < 2:
            return JSONResponse(
                status_code=400,
                content={
                    "status": 400,
                    "msg": "Envie ao menos 2 imagens para cadastro.",
                }
            )

        valid_embs = []
        debug_infos = []

        # -------------------------------------------
        # 3) Processar cada imagem do ENROLL
        # -------------------------------------------
        for idx, img_b64 in enumerate(body.images_base64, start=1):
            # decodificar
            try:
                bgr = decode_base64_to_bgr(img_b64)
            except Exception as e:
                debug_infos.append({"index": idx, "msg": f"Base64 inválido: {str(e)}"})
                continue

            # detectar rosto + embedding
            emb, face = get_best_face_embedding(bgr)
            if emb is None or face is None:
                debug_infos.append({"index": idx, "msg": "Nenhum rosto detectado"})
                continue

            # gerar face_box
            x1, y1, x2, y2 = map(int, face.bbox.tolist())
            face_box = (x1, y1, x2, y2)

            # anti-spoof híbrido atualizado
            label_spoof, prob_real_cnn, phone_score, glare_score, prob_adj = classify_spoof_hybrid(
                bgr, face_box, debug=False
            )

            debug_infos.append({
                "index": idx,
                "spoof_label": label_spoof,
                "prob_real_cnn": prob_real_cnn,
                "phone_score": phone_score,
                "glare_score": glare_score,
                "prob_real_adj": prob_adj,
            })

            # aceitar só imagens REAL
            if label_spoof != "REAL":
                continue

            # acumular embedding
            valid_embs.append(emb)

        # -------------------------------------------
        # 4) Precisa ter pelo menos 2 imagens REAL
        # -------------------------------------------
        if len(valid_embs) < 2:
            return JSONResponse(
                status_code=400,
                content={
                    "status": 400,
                    "msg": "Menos de 2 imagens válidas (REAL). Cadastro rejeitado.",
                    "debug": debug_infos,
                }
            )

        # -------------------------------------------
        # 5) Média dos embeddings (igual ao Colab)
        # -------------------------------------------
        mean_emb = np.mean(np.stack(valid_embs, axis=0), axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)

        # salvar no Firebase
        save_face_embedding(uid, mean_emb.tolist())

        return JSONResponse(
            status_code=201,
            content={
                "status": 201,
                "msg": "Face cadastrada com sucesso a partir de múltiplas imagens.",
                "uid": uid,
                "valid_images": len(valid_embs),
                "total_images": len(body.images_base64),
                "debug": debug_infos,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": 500, "msg": f"Erro interno ao cadastrar face: {str(e)}"}
        )


# ============================================
# Reconhecimento Facial - Verificação (VERIFY)
# ============================================
@app.post(
    "/verify-face",
    tags=["Reconhecimento Facial"],
    summary="Verificar face do cliente (comparar com embedding salvo)",
    response_model=VerifyFaceResponse,
)
async def verify_face(body: VerifyFaceRequest):
    try:
        # 1) valida idToken -> uid
        try:
            uid = verify_id_token(body.idToken)
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail={"status": 401, "msg": f"idToken inválido: {str(e)}"}
            )

        # 2) carrega embedding de referência do Firestore
        db_emb_list = get_face_embedding(uid)
        if db_emb_list is None:
            return VerifyFaceResponse(
                status=404,
                msg="Usuário não possui face cadastrada.",
                spoof_label="UNKNOWN",
                prob_real_cnn=0.0,
                phone_score=0.0,
                glare_score=0.0,
                prob_real_adj=0.0,
                similarity=0.0,
                passed=False,
            )

        db_emb = np.array(db_emb_list, dtype=np.float32)
        db_emb = db_emb / (np.linalg.norm(db_emb) + 1e-10)

        # 3) decodifica imagem
        bgr = decode_base64_to_bgr(body.image_base64)

        # 4) anti-spoof
        emb_try, face = get_best_face_embedding(bgr)
        if emb_try is None or face is None:
            return VerifyFaceResponse(
                status=200,
                msg="Nenhum rosto detectado.",
                spoof_label="UNKNOWN",
                prob_real_cnn=0.0,
                phone_score=0.0,
                glare_score=0.0,
                prob_real_adj=0.0,
                similarity=0.0,
                passed=False,
            )

        x1, y1, x2, y2 = map(int, face.bbox.tolist())
        face_box = (x1, y1, x2, y2)

        label_spoof, prob_real_cnn, phone_score, glare_score, prob_real_adj = classify_spoof_hybrid(
            bgr, face_box, debug=False
        )

        if label_spoof != "REAL":
            return VerifyFaceResponse(
                status=200,
                msg="Anti-spoof reprovou a imagem.",
                spoof_label=label_spoof,
                prob_real_cnn=prob_real_cnn,
                phone_score=phone_score,
                glare_score=glare_score,
                prob_real_adj=prob_real_adj,
                similarity=0.0,
                passed=False,
            )

        # 6) similaridade com embedding salvo
        sim = float(np.dot(emb, db_emb))
        passed = (label_spoof == "REAL") and (sim >= FACE_THRESHOLD)

        return VerifyFaceResponse(
            status=200,
            msg="Verificação realizada.",
            spoof_label=label_spoof,
            prob_real_cnn=prob_real_cnn,
            phone_score=phone_score,
            glare_score=glare_score,
            prob_real_adj=prob_real_adj,
            similarity=sim,
            passed=passed,
        )

    except HTTPException:
        raise
    except Exception as e:
        # se quiser, pode trocar para JSONResponse simples
        return VerifyFaceResponse(
            status=500,
            msg=f"Erro interno ao verificar face: {str(e)}",
            spoof_label="ERROR",
            prob_real_cnn=0.0,
            phone_score=0.0,
            glare_score=0.0,
            prob_real_adj=0.0,
            similarity=0.0,
            passed=False,
        )


# ============================================
# Reconhecimento Facial - Listagem (debug)
# ============================================
@app.get(
    "/list-face",
    tags=["Reconhecimento Facial"],
    summary="Consultar se o usuário possui face cadastrada (usa uid como id_cliente)",
)
async def list_face(id_cliente: str = Query(..., description="uid do usuário no Firebase")):
    """
    Endpoint simples de debug:
    - recebe id_cliente (uid do Firebase)
    - retorna se há embedding salvo no Firestore
    """
    try:
        emb = get_face_embedding(id_cliente)
        if emb is None:
            return JSONResponse(
                status_code=404,
                content={"status": 404, "msg": f"Cliente '{id_cliente}' não possui face cadastrada."}
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "msg": "Face encontrada para este cliente.",
                "embedding_dim": len(emb)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": 500, "msg": f"Erro interno ao consultar face: {str(e)}"}
        )