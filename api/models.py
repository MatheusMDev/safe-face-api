# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class RegisterFaceRequest(BaseModel):
    name: str
    # você pode adicionar outros campos, ex: email, id

class VerifyFaceRequest(BaseModel):
    name: str
    # usado para comparar rosto enviado com rosto cadastrado