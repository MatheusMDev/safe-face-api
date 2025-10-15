# ============================================
# Imports
# ============================================
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
# from api.models import RegisterFaceRequest, VerifyFaceRequest
# from api.utils import preprocess_image, compare_embeddings
# import tensorflow as tf
# from PIL import Image
import numpy as np


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

# Placeholder de “banco” de faces
# faces_db = {}

# model = tf.keras.models.load_model("app/face_model/model.h5") # futuro modelo


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
        # forçar falha a fim de testar exception
        # raise Exception("Falha simulada")
        return {"status": 200, "msg": "API On-line"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": 500, "msg": f"Erro interno: {str(e)}"}
        )    


# @app.post("/recognize")
# async def recognize_face(file: UploadFile = File(...)):
#     try:
#         # img = Image.open(file.file).convert("RGB")
#         # processed = preprocess_image(img)
#         # predictions = model.predict(processed)
#         return JSONResponse({"message": "API funcionando, imagem recebida."})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=400)

# @app.post("/register")
# async def register_face(data: RegisterFaceRequest, file: UploadFile = File(...)):
#     # Aqui você processaria a imagem e armazenaria embeddings
#     faces_db[data.name] = "embedding_placeholder"
#     return {"message": f"Rosto de {data.name} registrado."}

# @app.post("/verify")
# async def verify_face(data: VerifyFaceRequest, file: UploadFile = File(...)):
#     # Aqui você compararia a imagem com embeddings cadastrados
#     if data.name not in faces_db:
#         raise HTTPException(status_code=404, detail="Rosto não cadastrado")
#     # fake comparison
#     match = True
#     return {"match": match}


# ============================================
# Mock / Teste (remover depois)
# ============================================
## teste -> remover depois
faces_db = {
    "1": ["face_1", "face_2"],
    "2": ["face_3"]
}


# ============================================
# Reconhecimento Facial - Listagem
# ============================================
## lista faces do cliente com base no seu id (consulta fechada para não gargalar bd)
@app.get(
    "/list-faces",
    tags=["Reconhecimento Facial"],
    summary="Listar faces de um cliente",
    responses={
        200: {
            "description": "Faces encontradas",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "msg": "Faces encontradas com sucesso.",
                        "faces": ["face_1", "face_2"]
                    }
                }
            },
        },
        404: {
            "description": "Cliente não encontrado",
            "content": {
                "application/json": {
                    "example": {
                        "status": 404,
                        "msg": "Cliente 'cliente_99' não encontrado."
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
                            "value": {"status": 500, "msg": "Erro interno ao listar faces: Falha simulada"}
                        },
                        "excecao_desconhecida": {
                            "summary": "Exceção desconhecida",
                            "value": {"status": 500, "msg": "Erro interno ao listar faces: UnknownError"}
                        }
                    }
                }
            },
        },
    },
)
async def list_faces(id_cliente: str = Query(..., description="ID do Cliente p/ Consulta Fechada")):
    try:
        if id_cliente not in faces_db:
            return JSONResponse(
                status_code=404,
                content={"status": 404, "msg": f"Cliente '{id_cliente}' não encontrado."}
            )

        return JSONResponse(
            status_code=200,
            content={"status": 200, "msg": "Faces encontradas com sucesso.", "faces": faces_db[id_cliente]}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": 500, "msg": f"Erro interno ao listar faces: {str(e)}"}
        )
    

# @app.delete("/delete-face/{name}")
# async def delete_face(name: str):
#     if name in faces_db:
#         del faces_db[name]
#         return {"message": f"Rosto de {name} removido."}
#     else:
#         raise HTTPException(status_code=404, detail="Rosto não encontrado")
