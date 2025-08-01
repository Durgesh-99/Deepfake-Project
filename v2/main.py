from fastapi import FastAPI, UploadFile, HTTPException, Request,Body, File, Form
from fastapi.middleware.cors import CORSMiddleware
from database import user_collection
from pydantic import BaseModel, EmailStr
import torch.nn.functional as F
from models import UserAuth
from PIL import Image
import uvicorn
import hashlib
import torch
import io

from utils import Proto1, preprocess_image

app = FastAPI()

class ModelUpdateRequest(BaseModel):
    email: EmailStr
    model: str

class EmailRequest(BaseModel):
    email: EmailStr

# Enable CORS to allow access from any frontend or tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # restrict in production if desired
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Model Loading Function ---
def load_model(path="model_1.pth"):
    model = Proto1(img_size=256, num_classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Default model loaded at startup
default_model = load_model()

@app.get("/")
def read_root():
    return {"message": "FastAPI Deepfake Detection API is running."}

@app.post("/")
async def predict(
    image: UploadFile = File(...),
    DFD_SUBSCRIBE: str = Form(default="false")  # Get DFD from form data
):
    # Choose model based on DFD_SUBSCRIBE
    model_path = "model_1.pth" if DFD_SUBSCRIBE.lower() == "true" else "model_1.pth"
    model = load_model(model_path)

    # Read and process the image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(img)

    # Model inference and prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "class": int(predicted.item()),   # 0 = Fake, 1 = Real
        "confidence": float(confidence.item())
    }

@app.post("/get-model")
async def get_user_model(request: EmailRequest = Body(...)):
    user = user_collection.find_one({"email": request.email})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    model_version = user.get("model", "1_0")  # default fallback
    return {"email": request.email, "model": model_version}

@app.post("/subscribe")
async def update_model(req: ModelUpdateRequest):
    result = user_collection.update_one(
        {"email": req.email},
        {"$set": {"model": req.model}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": f"Model updated to {req.model}"}

@app.post("/auth")
async def auth(user: UserAuth):
    if user.action == "signup":
        if not user.name:
            raise HTTPException(status_code=422, detail="Name is required for signup")

        if user_collection.find_one({"email": user.email}):
            raise HTTPException(status_code=400, detail="User already exists")
        
        user_collection.insert_one({
            "name": user.name,
            "email": user.email,
            "password": hash_password(user.password),
            "model": "1_0"
        })
        return {"message": "User signed up successfully"}

    elif user.action == "login":
        found = user_collection.find_one({"email": user.email})
        if not found or found["password"] != hash_password(user.password):
            raise HTTPException(status_code=400, detail="Invalid email or password")

        return {"message": "Login successful"}

    else:
        raise HTTPException(status_code=400, detail="Invalid action")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
