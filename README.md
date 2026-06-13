# Vision AI Classification Platform

An end-to-end computer vision solution for image classification using EfficientNet-B0, PyTorch, and FastAPI. The system is trained on the CIFAR-10 dataset and provides real-time predictions through a REST API with an optional web-based frontend.

---

## Project Overview

This project demonstrates the complete machine learning lifecycle:

* Dataset preparation and preprocessing
* Transfer learning using EfficientNet-B0
* Model training and evaluation
* Inference on unseen images
* REST API deployment with FastAPI
* Frontend integration for image uploads
* Docker-based deployment support

The solution is designed to be lightweight, scalable, and production-ready.

---

## Dataset

**Dataset:** CIFAR-10

CIFAR-10 is a benchmark image classification dataset containing 60,000 RGB images distributed across 10 categories.

### Classes

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset Source:
https://www.cs.toronto.edu/~kriz/cifar.html

---

## Model Architecture

### EfficientNet-B0

EfficientNet-B0 was selected due to:

* Strong accuracy-to-parameter ratio
* Computational efficiency
* Proven transfer learning performance
* Fast inference speed
* Suitability for deployment environments

### Training Strategy

* Transfer Learning
* ImageNet Pretrained Weights
* Data Augmentation
* Train / Validation / Test Split
* Cross-Entropy Loss
* Adam Optimizer
* Learning Rate Scheduling

---

## Technology Stack

| Component     | Technology               |
| ------------- | ------------------------ |
| Language      | Python                   |
| Deep Learning | PyTorch                  |
| Model         | EfficientNet-B0          |
| API           | FastAPI                  |
| Frontend      | HTML, CSS, JavaScript    |
| Deployment    | Docker                   |
| Environment   | Google Colab / Local GPU |

---

## Project Structure

```text
vision-ai-classification-platform/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── class_names.json
│
├── frontend/
│   └── index.html
│
├── models/
│   └── efficientnet_b0_cifar10.pth
│
├── notebooks/
│   └── train_colab.ipynb
│
├── report/
│   └── technical_report.pdf
│
├── sample_images/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── train.py
```

---

## Model Training

### Google Colab

1. Open `notebooks/train_colab.ipynb`
2. Enable GPU:

   * Runtime → Change Runtime Type
   * Hardware Accelerator → GPU
3. Execute all notebook cells
4. Download generated model

Output:

```text
models/efficientnet_b0_cifar10.pth
```

### Local Training

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

---

## Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

Access Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /predict

Accepts an image and returns predicted class with confidence score.

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@sample_images/test_image.jpg"
```

#### Response

```json
{"prediction": "ship","confidence": 0.94}
```

---

## Frontend Interface

A lightweight frontend is included for interactive testing.

Features:

* Image Upload
* Real-Time Prediction
* Confidence Score Display
* API Integration

Launch frontend:

```bash
cd frontend
python -m http.server 5500
```

Open:

```text
http://localhost:5500
```

---

## Docker Deployment

Build Docker image:

```bash
docker build -t vision-ai-classification-platform .
```

Run container:

```bash
docker run -p 8000:8000 vision-ai-classification-platform
```

Access API:

```text
http://localhost:8000/docs
```

---

## Results

| Metric     | Value           |
| ---------- | --------------- |
| Model      | EfficientNet-B0 |
| Dataset    | CIFAR-10        |
| Classes    | 10              |
| Input Size | 224 × 224       |
| Framework  | PyTorch         |

*Actual performance metrics may vary depending on training configuration and hardware.*

---


