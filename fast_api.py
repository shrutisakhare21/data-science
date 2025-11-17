from pydantic import BaseModel
import os
import numpy as np
import pickle
import requests
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis

app = FastAPI(title="Face Authentication Service", description="Face verification using InsightFace", version="1.0")


face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))


DB_FILE = "embeddings.pkl"

def load_db():
    if os.path.exists(DB_FILE):
        return pickle.load(open(DB_FILE, "rb"))
    return {}

def save_db(db):
    pickle.dump(db, open(DB_FILE, "wb"))


def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


def extract_embedding(image_url):
    img = load_image_from_url(image_url)
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return faces[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class RegisterRequest(BaseModel):
    user_id: str
    image_url: str

class AuthRequest(BaseModel):
    image_url: str
    threshold: float = 0.35


@app.post("/register")
def register_user(req: RegisterRequest):
    try:
        emb = extract_embedding(req.image_url)
        db = load_db()
        db[req.user_id] = emb
        save_db(db)
        return {"message": f"User '{req.user_id}' registered successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/authenticate")
def authenticate(req: AuthRequest):
    try:
        emb = extract_embedding(req.image_url)
        db = load_db()

        if not db:
            raise HTTPException(status_code=404, detail="No registered users found.")

        best_user = None
        best_score = -1

        for user_id, ref_emb in db.items():
            score = cosine_similarity(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_user = user_id

        result = {
            "best_match": best_user,
            "similarity_score": float(best_score),
            "authentication": "SUCCESS" if best_score >= req.threshold else "FAILED"
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
