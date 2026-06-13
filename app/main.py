from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from app.model import predict_image, CLASS_NAMES
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="FastAPI service that predicts the class of an uploaded image using EfficientNet-B0.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "Image Classification API is running", "classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        result = predict_image(image)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
