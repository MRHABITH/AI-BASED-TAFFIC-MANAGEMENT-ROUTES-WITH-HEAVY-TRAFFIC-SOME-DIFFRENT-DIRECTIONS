# Traffic Detection API - Hugging Face Spaces

This backend is optimized for deployment on Hugging Face Spaces.

## 📋 Prerequisites

- Hugging Face account (https://huggingface.co)
- YOLOv3 weights file (download from: https://pjreddie.com/media/files/yolov3.weights)

## 🚀 Deployment Steps

### 1. Prepare Your Space

1. Create a new Space on Hugging Face: https://huggingface.co/new-space
2. Select "Docker" as the SDK
3. Choose "Space name" and save

### 2. Upload Files to Your Space

Upload these files to your HF Space repository:
- `Dockerfile`
- `main.py`
- `requirements.txt`
- `yolov3.cfg`
- `yolov3.weights`
- `coco.names`

### 3. Environment Variables

Set in Space Settings → Secrets:
- `ALLOWED_ORIGINS`: `*` (or specific Vercel domain)
- `HOST`: `0.0.0.0`
- `PORT`: `8000`

### 4. Access Your API

Your API will be available at: `https://[username]-[spacename].hf.space/api/`

### 5. Update Frontend

Update `frontend/.env.production` with your HF Space URL:
```
VITE_API_BASE_URL=https://[username]-[spacename].hf.space/api
```

## 📚 API Endpoints

- `GET /` - API status
- `GET /api/health` - Health check
- `POST /api/detect` - Detect vehicles in images

## 🔧 Troubleshooting

**Issue**: "YOLOv3 weights file not found"
- **Solution**: Upload `yolov3.weights` file (236MB) to your Space

**Issue**: "CUDA not available"
- **Solution**: Normal on HF Spaces. Backend will use CPU (slower but works)

**Issue**: CORS errors
- **Solution**: Update `ALLOWED_ORIGINS` environment variable in Space Settings

## 📝 Notes

- First request may take time to load the model
- HF Spaces containers automatically restart after 48 hours of inactivity
- Free tier has CPU-only inference (GPU available with paid plan)
