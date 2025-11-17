from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
import pickle
import requests
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis

app = FastAPI(title="Face Authentication Service", version="1.0")

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

DB_FILE = "embeddings.pkl"

def load_db():
    if os.path.exists(DB_FILE):
        return pickle.load(open(DB_FILE, "rb"))
    return {}

def save_db(db):
    pickle.dump(db, open(DB_FILE, "wb"))

def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def extract_faces(image_url):
    img = load_image(image_url)
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    face_data = []
    for face in faces:
        face_data.append({
            "embedding": face.embedding,
            "bbox": [float(face.bbox[0]), float(face.bbox[1]),
                     float(face.bbox[2]), float(face.bbox[3])]
        })
    return face_data

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class VerifyRequest(BaseModel):
    image_url_1: str
    image_url_2: str
    threshold: float = 0.35

@app.post("/verify")
def verify_faces(req: VerifyRequest):
    try:
        faces1 = extract_faces(req.image_url_1)
        faces2 = extract_faces(req.image_url_2)

        face1 = faces1[0]
        face2 = faces2[0]

        score = cosine_similarity(face1["embedding"], face2["embedding"])
        result = "same person" if score >= req.threshold else "different person"

        return {
            "verification_result": result,
            "similarity_score": score,
            "threshold_used": req.threshold,
            "image_1_faces": {"count": len(faces1), "primary_face_bbox": face1["bbox"]},
            "image_2_faces": {"count": len(faces2), "primary_face_bbox": face2["bbox"]}
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Face Verification API", "endpoints": {"/verify": "POST two images for verification"}}
