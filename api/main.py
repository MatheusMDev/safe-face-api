from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from api.models import RegisterFaceRequest, VerifyFaceRequest
from api.utils import preprocess_image, compare_embeddings
# import tensorflow as tf
# from PIL import Image
import numpy as np

app = FastAPI(title="Face Recognition API", version="1.0")


# model = tf.keras.models.load_model("app/face_model/model.h5") # futuro modelo

@app.get("/health")
async def health():
    return {"status": "API online"}