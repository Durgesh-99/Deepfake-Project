from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn.functional as F

from utils import load_model, preprocess_image

app = FastAPI()

# Enable CORS to allow access from any frontend or tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # restrict in production if desired
    allow_methods=["*"],
    allow_headers=["*"]
)

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
