# 🔌 API DOCUMENTATION

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-backend.hf.space`

---

## Endpoints

### 1. Root Endpoint

```http
GET /
```

**Description**: Check if API is running

**Response (200 OK)**:
```json
{
  "message": "Traffic Detection API is running"
}
```

**Example**:
```bash
curl http://localhost:8000/
```

---

### 2. Health Check

```http
GET /health
```

**Description**: Health check endpoint for monitoring

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "model": "YOLOv3"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### 3. Vehicle Detection

```http
POST /api/detect
Content-Type: multipart/form-data
```

**Description**: Detect vehicles in uploaded images

**Request Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_1` | File | Optional | Image from road/intersection 1 |
| `image_2` | File | Optional | Image from road/intersection 2 |
| `image_3` | File | Optional | Image from road/intersection 3 |
| `image_4` | File | Optional | Image from road/intersection 4 |

**Constraints**:
- At least 1 image required
- Supported formats: JPEG, PNG
- Max size: 25 MB per image
- Recommended resolution: 640x480 or higher

**Response (200 OK)**:
```json
{
  "status": "success",
  "detections": [
    {
      "road": 1,
      "count": 5,
      "vehicles": ["car", "bus", "car", "truck", "car"],
      "confidence": 0.82
    },
    {
      "road": 2,
      "count": 3,
      "vehicles": ["car", "motorbike", "car"],
      "confidence": 0.78
    }
  ],
  "total_vehicles": 8
}
```

**Response Fields**:
- `status`: Request status ("success" or error)
- `detections`: Array of detection results for each image
  - `road`: Intersection/road number (1-4)
  - `count`: Total vehicles detected
  - `vehicles`: List of detected vehicle types
  - `confidence`: Average confidence score (0-1)
- `total_vehicles`: Total count across all images

**Error Responses**:

*400 Bad Request* - No images provided:
```json
{
  "detail": "No images provided"
}
```

*500 Internal Server Error* - Processing failed:
```json
{
  "detail": "Detection failed: [error message]"
}
```

---

## Examples

### JavaScript/Fetch

```javascript
async function detectVehicles(images) {
  const formData = new FormData();
  
  // Add up to 4 images
  images.forEach((image, index) => {
    formData.append(`image_${index + 1}`, image);
  });
  
  const response = await fetch(
    `${process.env.VITE_API_BASE_URL}/api/detect`,
    {
      method: 'POST',
      body: formData,
    }
  );
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return await response.json();
}

// Usage:
const fileInputs = document.querySelectorAll('input[type="file"]');
const images = Array.from(fileInputs)
  .map(input => input.files[0])
  .filter(Boolean);

detectVehicles(images)
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

### Python/Requests

```python
import requests

def detect_vehicles(image_paths):
    """
    Send images to backend for vehicle detection
    
    Args:
        image_paths: List of image file paths [image1, image2, ...]
    
    Returns:
        Dictionary with detection results
    """
    url = "http://localhost:8000/api/detect"
    
    files = {}
    for idx, image_path in enumerate(image_paths, 1):
        with open(image_path, 'rb') as f:
            files[f'image_{idx}'] = f
    
    response = requests.post(url, files=files)
    response.raise_for_status()
    
    return response.json()

# Usage:
result = detect_vehicles(['image1.jpg', 'image2.jpg'])
print(f"Total vehicles detected: {result['total_vehicles']}")

for detection in result['detections']:
    print(f"Road {detection['road']}: {detection['count']} vehicles")
    print(f"  Types: {', '.join(detection['vehicles'])}")
    print(f"  Confidence: {detection['confidence']:.2%}")
```

### cURL

```bash
# Single image
curl -X POST http://localhost:8000/api/detect \
  -F "image_1=@image1.jpg" \
  -H "Accept: application/json"

# Multiple images
curl -X POST http://localhost:8000/api/detect \
  -F "image_1=@image1.jpg" \
  -F "image_2=@image2.jpg" \
  -F "image_3=@image3.jpg" \
  -F "image_4=@image4.jpg"

# Pretty print JSON response
curl -X POST http://localhost:8000/api/detect \
  -F "image_1=@image1.jpg" | python -m json.tool
```

---

## Vehicle Types Detected

YOLO model detects the following classes:

```
- person
- bicycle
- car
- motorbike
- bus
- truck
- train
- truck
```

**Tracked classes** (in this API):
```
Vehicle Types = ["car", "bus", "motorbike", "truck", "bicycle", "person"]
```

---

## Performance Specifications

| Metric | Value |
|--------|-------|
| **Model** | YOLOv3 |
| **Input Resolution** | 416x416 |
| **Confidence Threshold** | 0.5 |
| **NMS Threshold** | 0.4 |
| **Avg Inference Time (CPU)** | 3-5 seconds per image |
| **Avg Inference Time (GPU)** | 0.5-1 second per image |
| **Memory Usage** | ~2GB (CPU) / ~6GB (GPU) |

---

## Error Handling

### Common Errors

**1. CORS Error**
```
Error: Access to XMLHttpRequest at 'https://api.example.com' 
from origin 'https://frontend.example.com' has been blocked by CORS policy
```

**Solution**: Ensure backend CORS is configured:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**2. 413 Payload Too Large**
```
HTTP 413: Request Entity Too Large
```

**Solution**: Reduce image file size
- Compress before upload
- Resize to 640x480 max
- Maximum 25 MB per image

**3. 504 Gateway Timeout**
```
HTTP 504: Gateway Timeout
```

**Solution**: 
- Reduce number of images (use 2 instead of 4)
- Upgrade to GPU backend
- Increase timeout in client (frontend)

---

## Rate Limiting (Optional)

Not currently implemented. To add rate limiting:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/detect")
@limiter.limit("5/minute")
async def detect_vehicles(...):
    # Max 5 requests per minute
    ...
```

---

## Authentication (Optional)

To add API key authentication:

```python
from fastapi import Header, HTTPException

@app.post("/api/detect")
async def detect_vehicles(
    api_key: str = Header(...),
    ...
):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    ...
```

---

## OpenAPI/Swagger Docs

Access interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These are automatically generated from the FastAPI docstrings.

---

## Webhook Support (Future Enhancement)

For async processing of large batches:

```python
@app.post("/api/detect-async")
async def detect_vehicles_async(
    webhook_url: str,
    images: List[UploadFile]
):
    """
    Process images asynchronously and send results to webhook
    """
    # Queue images for processing
    # When done, POST results to webhook_url
    ...
```

---

## SDK Support

### TypeScript/React (Frontend)

See `frontend/src/services/api.js` for pre-built client:

```javascript
import { detectVehicles } from './services/api';

const result = await detectVehicles([image1, image2, image3, image4]);
```

### Python

```bash
# Install SDK (if published)
pip install traffic-detection-sdk

# Use
from traffic_detection import Client
client = Client(api_url="http://localhost:8000")
result = client.detect_vehicles([image1, image2])
```

---

## Testing with Postman

1. Import collection from `backend/postman_collection.json` (if available)
2. Set environment variables:
   - `api_url`: `http://localhost:8000`
3. Run requests in sequence

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-03-13 | Initial release |
| - | - | - |

---

## Support

For issues or questions:
- Check logs: `http://backend:8000/logs`
- Review tests: `test_project.py`
- Contact: [your-email]@example.com
