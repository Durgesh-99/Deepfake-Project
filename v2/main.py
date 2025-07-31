from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn.functional as F

from utils import Proto1, preprocess_image

app = FastAPI()

# Enable CORS to allow access from any frontend or tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # restrict in production if desired
    allow_methods=["*"],
    allow_headers=["*"]
)

'''
# Load the trained model once at startup
model = load_model()

@app.get("/")
def read_root():
    return {"message": "FastAPI Deepfake Detection API is running."}

@app.post("/")
async def predict(image: UploadFile = File(...)):
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
@app.post("/predict-batch")
async def predict_batch(images: list[UploadFile] = File(...)):
    results = []
    for image_file in images:
        try:
            contents = await image_file.read()
            img  = Image.open(io.BytesIO(contents)).convert("RGB")
            input_tensor = preprocess_image(img)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim =1)
                confidance, predicted = torch.max(probs,1)
            results.append({
                "filename" : image_file.filename,
                "class" : int(predicted.item()),
                "confidence" : float(confidance.item())

            })
        except Exception as e:
            results.append({
                "filename"  : image_file.filename,
                "error" : str(e)
            })
    return results
'''

# --- Model Loading Function ---
def load_model(path="model_1.pth"):
    model = Proto1(img_size=256, num_classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

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

@app.post("/predict-batch")
async def predict_batch(
    images: list[UploadFile] = File(...),
    DFD_SUBSCRIBE: str = Form(default="false")
):
    model_path = "model_1.pth" if DFD_SUBSCRIBE.lower() == "true" else "model_1.pth"
    model = load_model(model_path)

    results = []
    for image_file in images:
        try:
            contents = await image_file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            input_tensor = preprocess_image(img)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
            results.append({
                "filename": image_file.filename,
                "class": int(predicted.item()),
                "confidence": float(confidence.item())
            })
        except Exception as e:
            results.append({
                "filename": image_file.filename,
                "error": str(e)
            })
    return results

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
