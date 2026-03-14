FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and YOLOv10
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (ultralytics will auto-download YOLOv10 models)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .

# Pre-download YOLOv10 model on build (optional, speeds up startup)
RUN python -c "from ultralytics import YOLO; YOLO('yolov10m.pt')" || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
