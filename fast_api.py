from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
from typing import List, Dict

app = FastAPI(
    title="Face Authentication Service",
    description="Face verification using InsightFace",
    version="1.0"
)


face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))


class VerifyRequest(BaseModel):
    image_url_1: str
    image_url_2: str
    threshold: float = 0.35


def load_image_from_url(url: str):
    """Load image from URL and convert to numpy array"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def extract_faces(image_url: str) -> List[Dict]:
    """Extract all faces from an image with embeddings and bounding boxes"""
    img = load_image_from_url(image_url)
    faces = face_app.get(img)
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    
    face_data = []
    for face in faces:
        face_data.append({
            "embedding": face.embedding,
            "bbox": {
                "x1": float(face.bbox[0]),
                "y1": float(face.bbox[1]),
                "x2": float(face.bbox[2]),
                "y2": float(face.bbox[3])
            },
            "confidence": float(face.det_score)
        })
    
    return face_data

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.post("/verify")
def verify_faces(req: VerifyRequest):
    """
    Compare two face images and determine if they're the same person.
    
    Returns:
    - verification_result: "same person" or "different person"
    - similarity_score: float between -1 and 1
    - faces_detected: information about detected faces including bounding boxes
    """
    try:
        # Extract faces from both images
        faces_1 = extract_faces(req.image_url_1)
        faces_2 = extract_faces(req.image_url_2)
        
        # Use the first (most confident) face from each image
        face_1 = faces_1[0]
        face_2 = faces_2[0]
        
        # Calculate similarity
        similarity_score = cosine_similarity(face_1["embedding"], face_2["embedding"])
        
        # Determine verification result
        verification_result = "same person" if similarity_score >= req.threshold else "different person"
        
        return {
            "verification_result": verification_result,
            "similarity_score": similarity_score,
            "threshold_used": req.threshold,
            "image_1_faces": {
                "count": len(faces_1),
                "primary_face": {
                    "bounding_box": face_1["bbox"],
                    "confidence": face_1["confidence"]
                }
            },
            "image_2_faces": {
                "count": len(faces_2),
                "primary_face": {
                    "bounding_box": face_2["bbox"],
                    "confidence": face_2["confidence"]
                }
            }
        }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Face Verification API",
        "endpoints": {
            "/verify": "POST - Verify if two face images are the same person"
        }
    }
