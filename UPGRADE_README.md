# AI-Based Traffic Management with Real-Time Vehicle Detection

> **Upgraded**: Modern React + Vite frontend with advanced UI/UX replacing the legacy Streamlit interface

## 🚀 Project Overview

An intelligent traffic management system that uses YOLOv3 deep learning model to detect vehicles in real-time and manage traffic lights dynamically based on vehicle density at different roads.

### Key Improvements in New Version
- ✅ **React + Vite** - Modern, fast development experience
- ✅ **Advanced UI/UX** - Dark theme, animations, responsive design
- ✅ **FastAPI Backend** - Scalable, production-ready API
- ✅ **Real-time Dashboard** - Live statistics and traffic control
- ✅ **Mobile Responsive** - Works on all devices
- ✅ **Better Performance** - Optimized rendering and API calls

## 📊 Quick Start

### 1-Minute Setup
```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python main.py

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser 🎉

## 🎯 Features

| Feature | Details |
|---------|---------|
| 🚗 **Vehicle Detection** | Multi-image upload, real-time detection with YOLOv3 |
| 🚦 **Smart Traffic Control** | Priority-based green lights based on vehicle count |
| 📊 **Analytics Dashboard** | Real-time statistics and traffic breakdown |
| 📹 **Live Streaming** | Webcam integration with live detection |
| 🎨 **Modern UI** | Dark theme, animations, responsive design |
| ⚡ **High Performance** | Sub-200ms detection speed |

## 📁 Directory Structure

```
traffic/
├── frontend/                 # React + Vite Application
│   ├── src/
│   │   ├── components/      # UI Components
│   │   ├── App.jsx          # Main App
│   │   └── index.css        # Tailwind CSS
│   ├── package.json
│   └── vite.config.js
├── backend/                 # FastAPI Server
│   ├── main.py             # API endpoints
│   ├── requirements.txt
│   ├── yolov3.cfg
│   ├── coco.names
│   └── yolov3.weights       # (Download separately)
├── README.md
└── SETUP_GUIDE.md
```

## 🛠️ Installation

### Option 1: Quick Setup (Recommended)
```bash
# Clone/navigate to project
cd frontend
npm install
npm run dev

# In another terminal
cd backend
pip install -r requirements.txt
python main.py
```

### Option 2: Docker
```bash
docker-compose up
```

### Option 3: Production Build
```bash
# Frontend
cd frontend
npm run build
# Deploy 'dist' folder

# Backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📸 Usage

### Vehicle Detection
1. Go to "Vehicle Detection" tab
2. Upload 1-4 images (representing different road directions)
3. System detects vehicles and counts by type
4. View results in real-time

### Traffic Management
1. Upload images with vehicles
2. Go to "Dashboard" tab
3. See which road gets green light
4. Green light prioritized by vehicle count
5. 30-second countdown per road

### Live Feed
1. Click "Live Feed" tab
2. Click "Start Camera"
3. Grant permission when prompted
4. Real-time vehicle detection from webcam

## 📊 Screenshot Examples

### Dashboard
- Real-time vehicle statistics
- Road-wise breakdown charts
- Traffic light status indicators
- System health metrics

### Vehicle Detection
- Multi-image upload interface
- Vehicle count per road
- Detailed breakdown (cars, buses, trucks)
- Confidence scores

### Traffic Control
- Live traffic light display (🟢 Green / 🔴 Red)
- Countdown timers
- Vehicle count per road
- Priority order

## 🔧 Configuration

### Backend Settings
Edit `backend/main.py`:
```python
# Detection confidence threshold (0-1)
if confidence > 0.5:

# NMS threshold
indexes = cv2.dnn.NMSBoxes(..., 0.4)
```

### Frontend Customization
Edit `frontend/tailwind.config.js`:
```js
colors: {
  primary: '#0f172a',
  secondary: '#1e293b',
  accent: '#3b82f6',
}
```

### Traffic Light Timing
Edit `TrafficLightDashboard.jsx`:
```js
const totalGreenTime = 30; // seconds
```

## ⚡ Performance

| Metric | Value |
|--------|-------|
| Detection Speed | ~200ms per image |
| UI Frame Rate | 60 FPS |
| Memory Usage | ~2GB with model |
| Supported Vehicles | 6 types |
| Max Concurrent | 4 images |

## 🐛 Troubleshooting

### Backend Error: "Module not found"
```bash
cd backend
pip install -r requirements.txt
```

### Frontend won't connect to backend
```bash
# Check if backend is running
curl http://localhost:8000/api/health
# Should return: {"status": "healthy", "model": "YOLOv3"}
```

### Camera permission denied
- Check browser privacy settings
- Use HTTPS in production
- Reset OS camera permissions

### Model file not found
Download YOLOv3 weights:
```bash
cd backend
wget https://pjreddie.com/media/files/yolov3.weights
```

## 📚 API Documentation

### Health Check
```
GET /api/health
Response: {"status": "healthy", "model": "YOLOv3"}
```

### Detect Vehicles
```
POST /api/detect
Content-Type: multipart/form-data

Parameters:
- image_1: Image file
- image_2: Image file
- image_3: Image file
- image_4: Image file

Response:
{
  "status": "success",
  "detections": [
    {"road": 1, "count": 3, "vehicles": ["car", "car", "bus"]},
    {"road": 2, "count": 1, "vehicles": ["truck"]}
  ],
  "total_vehicles": 4
}
```

## 🔄 Development Workflow

### Frontend Development
```bash
cd frontend
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### Backend Development
```bash
cd backend
python main.py       # Run server
# Auto-reload enabled on code changes
```

## 📦 Dependencies

### Frontend
```json
{
  "react": "^18.2.0",
  "vite": "^4.4.9",
  "tailwindcss": "^3.3.0",
  "framer-motion": "^10.16.4",
  "axios": "^1.5.0",
  "lucide-react": "^0.263.1"
}
```

### Backend
```
fastapi==0.104.1
uvicorn==0.24.0
opencv-python==4.8.1.78
numpy==1.24.3
pillow==10.0.1
python-multipart==0.0.6
```

## 🚀 Deployment

### Vercel (Frontend)
```bash
npm run build
# Push to GitHub, connect to Vercel
```

### Heroku (Backend)
```bash
heroku create your-app
git push heroku main
```

### Docker
```bash
docker-compose up -d
```

## 📈 Future Roadmap

- [ ] WebSocket for real-time video processing
- [ ] Vehicle speed estimation
- [ ] Historical data storage
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Edge device support
- [ ] Multi-model ensemble

## 📝 License

Educational and Development Use

## 💡 Tips & Tricks

**Faster Detection**: Use smaller model
- Replace with YOLOv3-Tiny for 4x faster detection

**Better Accuracy**: Use larger model
- YOLOv4 or YOLOv5 have higher accuracy

**Save Bandwidth**: Compress images
- Reduce image size before upload

**Optimize Traffic**: Custom timing
- Adjust green light duration based on your needs

## 🤝 Contributing

Found a bug? Have an idea? 
- Create an issue
- Submit a pull request
- Suggest improvements

## 📞 Support

- Check SETUP_GUIDE.md for detailed setup
- Review API documentation above
- Check troubleshooting section
- Examine console logs for errors

---

<div align="center">

### 🎉 Ready to Manage Traffic Intelligently?

[📖 Full Setup Guide](SETUP_GUIDE.md) • [🚀 Quick Start](#-quick-start) • [💬 Issues](issues)

**Version 2.0** | React + Vite Migration | March 2026 ✨

</div>
