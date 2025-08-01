from fastapi import FastAPI, UploadFile, HTTPException, Depends, Request,Body, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from database import user_collection
from pydantic import BaseModel, EmailStr
import torch.nn.functional as F
from models import UserAuth
from PIL import Image
import uvicorn
import hashlib
import torch
import io
from jose import JWTError, jwt
from datetime import datetime, timedelta

from utils import Proto1, preprocess_image

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")

app = FastAPI()

class ModelUpdateRequest(BaseModel):
    email: EmailStr
    model: str

class EmailRequest(BaseModel):
    email: EmailStr

SECRET_KEY = "your_super_secret_key"  # Use env var in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Enable CORS to allow access from any frontend or tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://127.0.0.1:5500"], # restrict in production if desired
    allow_credentials=True,
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
    request: Request,
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
async def get_user_model(current_user: str = Depends(verify_token)):
    user = user_collection.find_one({"email": current_user})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    model_version = user.get("model", "1_0")  # default fallback
    return {"email": current_user, "model": model_version}

@app.post("/subscribe")
async def update_model(
    req: ModelUpdateRequest = Body(...),
    current_user: str = Depends(verify_token)
):
    result = user_collection.update_one(
        {"email": current_user},
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
        token = create_access_token({"sub": user.email})
        return {"message": "User signed up successfully", "token": token}


    elif user.action == "login":
        found = user_collection.find_one({"email": user.email})
        if not found or found["password"] != hash_password(user.password):
            raise HTTPException(status_code=400, detail="Invalid email or password")

        token = create_access_token({"sub": user.email})
        return {"message": "Login successfully", "token": token}


    else:
        raise HTTPException(status_code=400, detail="Invalid action")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
