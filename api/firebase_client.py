# firebase_client.py
import time
from typing import Optional, List

import firebase_admin
from firebase_admin import credentials, firestore, auth

from config import FIREBASE_CREDENTIALS, FIREBASE_FACES_COLLECTION

# Inicializa Firebase Admin
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred)

db = firestore.client()


def verify_id_token(id_token: str) -> str:
    """
    Verifica o idToken do Firebase e devolve o uid do usuário.
    Lança exception se inválido.
    """
    decoded = auth.verify_id_token(id_token)
    uid = decoded["uid"]
    return uid


def save_face_embedding(uid: str, embedding: List[float]):
    """
    Salva/atualiza o embedding de face do usuário no Firestore.
    """
    doc_ref = db.collection(FIREBASE_FACES_COLLECTION).document(uid)
    doc_ref.set({
        "embedding": embedding,
        "created_at": firestore.SERVER_TIMESTAMP
    })


def get_face_embedding(uid: str) -> Optional[List[float]]:
    """
    Retorna o embedding da face do usuário ou None se não existir.
    """
    doc_ref = db.collection(FIREBASE_FACES_COLLECTION).document(uid)
    doc = doc_ref.get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    return data.get("embedding")