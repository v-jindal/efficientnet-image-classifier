import json
from pathlib import Path
import torch
import timm
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_cifar10.pth"
CLASS_PATH = Path(__file__).resolve().parent / "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def build_model(num_classes: int = 10):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=num_classes
    )
    return model

_model = None

def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}"
            )

        model = build_model(len(CLASS_NAMES))

        state = torch.load(MODEL_PATH, map_location=DEVICE)

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

        _model = model

    return _model

def predict_image(image):
    model = load_model()

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return {
        "prediction": CLASS_NAMES[predicted_idx.item()],
        "confidence": round(confidence.item(), 4)
    }