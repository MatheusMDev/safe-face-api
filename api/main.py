from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
# from api.models import RegisterFaceRequest, VerifyFaceRequest
# from api.utils import preprocess_image, compare_embeddings
# import tensorflow as tf
# from PIL import Image
import numpy as np

app = FastAPI(title="Face Recognition API", version="1.0")

# Placeholder de “banco” de faces
# faces_db = {}

# model = tf.keras.models.load_model("app/face_model/model.h5") # futuro modelo

@app.get("/health")
async def health():
    return {"status": "API online"}

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

# @app.get("/list-faces")
# async def list_faces():
#     return {"faces": list(faces_db.keys())}

# @app.delete("/delete-face/{name}")
# async def delete_face(name: str):
#     if name in faces_db:
#         del faces_db[name]
#         return {"message": f"Rosto de {name} removido."}
#     else:
#         raise HTTPException(status_code=404, detail="Rosto não encontrado")
