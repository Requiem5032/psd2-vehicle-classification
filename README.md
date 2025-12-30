# Vehicle Sensor Classification

This repository contains code and data for classifying vehicle types (bike, bus, car) using sensor data and neural network models.

## Project Structure

```
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints and configuration
│   ├── predictor.py       # Model loading and prediction logic
│   └── schemas.py         # Pydantic request/response models
├── src/                   # Source code for models and utilities
│   ├── nn/               # Neural network models and dataloaders
│   └── utils/            # Data processing and evaluation utilities
├── notebooks/            # Jupyter notebooks for analysis and training
├── data/                 # Raw and processed sensor data
├── results/              # Model training results and saved models
├── Dockerfile            # Container configuration
├── requirements-api.txt  # API dependencies
└── run.py               # CLI script for running predictions
```

## Quick Start

### Create and Activate Conda Environment

```bash
conda env create -f environment.yaml
conda activate psd2
```

### CLI Usage

```bash
python run.py --help
python run.py results/convolutional_neural_network/averaged_model/best_model.pth data/bike.csv
```

---

## REST API

The API provides endpoints for vehicle transportation mode classification from accelerometer sensor data.

### Running the API Locally

```bash
# Install dependencies
pip install -r requirements-api.txt

# Run with default settings (2-second windows)
uvicorn api.main:app --host 0.0.0.0 --port 8080

# Run with custom window size (30 seconds)
WINDOW_SECONDS=30.0 uvicorn api.main:app --host 0.0.0.0 --port 8080

# Run with auto-reload for development
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### API Documentation

Once running, access the interactive documentation:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json

### Endpoints

#### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "window_seconds": 30.0
}
```

#### Predict Transportation Mode

```bash
POST /predict
Content-Type: multipart/form-data
```

**Request:** Upload a CSV file with accelerometer data.

**Required CSV Columns:**
| Column | Description |
|--------|-------------|
| `time` | Timestamp in seconds |
| `ax` | Accelerometer X-axis reading |
| `ay` | Accelerometer Y-axis reading |
| `az` | Accelerometer Z-axis reading |

**Response:**
```json
{
  "status": "success",
  "window_seconds": 30.0,
  "total_segments": 5,
  "overall_prediction": {
    "label": "bus",
    "label_id": 1,
    "confidence": 0.8
  },
  "segment_predictions": [
    {
      "segment_id": 0,
      "time_start": 0.0,
      "time_end": 30.0,
      "label": "bus",
      "label_id": 1,
      "probabilities": {
        "bike": 0.05,
        "bus": 0.90,
        "car": 0.05
      }
    }
  ],
  "mode_distribution": {
    "bike": {"count": 0, "percentage": 0.0},
    "bus": {"count": 4, "percentage": 80.0},
    "car": {"count": 1, "percentage": 20.0}
  }
}
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# Predict from CSV file
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/bus.csv"

# Pretty print response
curl -X POST http://localhost:8080/predict \
  -F "file=@data/bus.csv" | python -m json.tool

# Test with different vehicle types
curl -X POST http://localhost:8080/predict -F "file=@data/bike.csv"
curl -X POST http://localhost:8080/predict -F "file=@data/car-clear.csv"
curl -X POST http://localhost:8080/predict -F "file=@data/rail.csv"
```

### Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `WINDOW_SECONDS` | Duration of each analysis window in seconds | `2.0` |
| `PORT` | Server port | `8080` |

### Label Mapping

| Label ID | Label Name |
|----------|------------|
| 0 | bike |
| 1 | bus |
| 2 | car |

---

## Docker

### Build Image

```bash
docker build -t vehicle-classification-api .
```

### Run Container

```bash
# Default settings
docker run -p 8080:8080 vehicle-classification-api

# Custom window size
docker run -p 8080:8080 -e WINDOW_SECONDS=30.0 vehicle-classification-api

# With volume mount for custom model
docker run -p 8080:8080 \
  -v /path/to/model:/app/results/convolutional_neural_network/averaged_model \
  vehicle-classification-api
```

---

## Deploy to Google Cloud Run

### Prerequisites

```bash
# Install Google Cloud SDK

# Login
gcloud auth login
firebase login
```

### Deploy

```bash
# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/vehicle-classification-api

gcloud run deploy vehicle-classification-api \
  --image gcr.io/$PROJECT_ID/vehicle-classification-api \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars "WINDOW_SECONDS=30.0"
```

### Get Service URL

```bash
gcloud run services describe vehicle-classification-api \
  --region asia-southeast1 \
  --format 'value(status.url)'
```

---

## CSV Format Example

```csv
time,gFx,gFy,gFz,ax,ay,az,wx,wy,wz,...
0.078756,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,...
0.123827,-0.0488,0.3217,1.0039,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,...
```

**Note:** Only `time`, `ax`, `ay`, `az` columns are required. Additional columns are ignored.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Invalid file type (not CSV) |
| 400 | Missing required columns |
| 400 | Insufficient data |
| 400 | Empty CSV file |
| 500 | Model not available |
| 500 | Prediction failed |

**Error Response Format:**
```json
{
  "status": "error",
  "message": "Invalid CSV data",
  "details": "Missing required columns: ax, ay"
}
```

---

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 3 channels (ax, ay, az) x 50 samples per segment
- **Output**: 3 classes (bike, bus, car)
- **Trained Model**: `results/convolutional_neural_network/averaged_model/best_model.pth`
