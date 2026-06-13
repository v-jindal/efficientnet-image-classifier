# Image Classification System with FastAPI

This project trains an EfficientNet-B0 image classification model on the CIFAR-10 dataset and exposes predictions through a FastAPI endpoint.

## Dataset

Dataset used: CIFAR-10

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

CIFAR-10 has 60,000 color images across 10 classes. It satisfies the assignment requirement of at least 5 classes.

## Model

Model selected: EfficientNet-B0 with ImageNet pretrained weights.

Reason: EfficientNet-B0 is lightweight, accurate, and suitable for transfer learning. It gives good performance without requiring very high GPU resources.

## Project Structure

```text
image_classification_fastapi_project/
├── app/
│   ├── main.py
│   ├── model.py
│   └── class_names.json
├── models/
│   └── efficientnet_b0_cifar10.pth   # generated after training
├── notebooks/
│   └── train_colab.ipynb
├── train.py
├── requirements.txt
├── Dockerfile
├── sample_request_response.txt
└── report/technical_report.pdf
```

## How to Train the Model

### Option 1: Google Colab

1. Open `notebooks/train_colab.ipynb` in Google Colab.
2. Go to `Runtime > Change runtime type > GPU`.
3. Run all cells.
4. Download the generated model file:

```text
models/efficientnet_b0_cifar10.pth
```

5. Put this file inside the `models/` folder of this project.

### Option 2: Local Training

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Open this in browser:

```text
http://127.0.0.1:8000/docs
```

Upload an image using the `/predict` endpoint.

## Sample API Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_images/test_image.jpg"
```

## Sample API Response

```json
{
  "prediction": "ship",
  "confidence": 0.94
}
```

## Docker Instructions

Build the Docker image:

```bash
docker build -t image-classification-api .
```

Run the container:

```bash
docker run -p 8000:8000 image-classification-api
```

## Submission Checklist

Submit these items:

- GitHub repository link
- README.md
- Trained model file: `models/efficientnet_b0_cifar10.pth`
- Sample API request and response
- Technical report PDF: `report/technical_report.pdf`
