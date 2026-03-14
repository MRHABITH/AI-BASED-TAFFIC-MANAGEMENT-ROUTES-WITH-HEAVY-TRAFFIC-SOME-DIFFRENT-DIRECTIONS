# Complete Installation Guide

Follow these steps to set up the AI-Based Traffic Management System with React + Vite frontend.

## Prerequisites Checklist

Before you begin, ensure you have:
- [ ] Windows, macOS, or Linux operating system
- [ ] Administrator/sudo access (for some installations)
- [ ] 4GB+ RAM available
- [ ] 2GB+ free disk space
- [ ] Stable internet connection

## Step 1: Install Required Software

### 1.1 Install Node.js (for Frontend)

**Windows & macOS:**
1. Visit https://nodejs.org/
2. Download **LTS version** (18.x or later)
3. Run the installer
4. Follow the installation wizard
5. Restart your computer
6. Verify installation:
   ```bash
   node --version
   npm --version
   ```

**Linux (Ubuntu/Debian):**
```bash
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 1.2 Install Python (for Backend)

**Windows:**
1. Visit https://python.org/downloads/
2. Download Python 3.10 or 3.11 (with pip)
3. Run installer
4. ⚠️ **IMPORTANT**: Check "Add Python to PATH"
5. Click "Install Now"
6. Restart your computer
7. Verify in Terminal/PowerShell:
   ```bash
   python --version
   pip --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y python3.10 python3-pip
```

### 1.3 Install Git (Optional but Recommended)

**Windows:** https://git-scm.com/download/win
**macOS:** `brew install git`
**Linux:** `sudo apt install -y git`

## Step 2: Clone/Navigate to Project

### If You Don't Have Git:
Simply navigate to your project directory.

### If You Have Git:
```bash
git clone <repository-url>
cd AI-BASED-TAFFIC-MANAGEMENT-ROUTES-WITH-HEAVY-TRAFFIC-SOME-DIFFRENT-DIRECTIONS
```

## Step 3: Backend Setup

### 3.1 Navigate to Backend Directory
```bash
cd backend
```

### 3.2 Create Python Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

### 3.3 Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- OpenCV (computer vision)
- NumPy (numerical arrays)
- Python-multipart (file uploads)
- Uvicorn (ASGI server)

**Installation time:** 2-5 minutes

### 3.4 Download YOLOv3 Model Files

**Critical:** Download the pre-trained YOLOv3 weights (236 MB)

**Option A: Using Command Line**
```bash
# Windows
powershell -Command "Invoke-WebRequest -Uri 'https://pjreddie.com/media/files/yolov3.weights' -OutFile 'yolov3.weights'"

# macOS/Linux
wget https://pjreddie.com/media/files/yolov3.weights
```

**Option B: Manual Download**
1. Visit https://pjreddie.com/darknet/yolo/
2. Look for "Download YOLOv3 weights"
3. Download to `backend/` directory
4. File should be named: `yolov3.weights` (236 MB)

**Verify:** After download, you should have these files in `backend/`:
```
yolov3.weights    (236 MB)
yolov3.cfg        (125 KB)
coco.names        (1 KB)
```

### 3.5 Test Backend Setup

```bash
python main.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Uvicorn server is running
```

Open browser: http://localhost:8000/api/health
Should show: `{"status": "healthy", "model": "YOLOv3"}`

**If it fails**, see Troubleshooting section below.

## Step 4: Frontend Setup

### 4.1 Open New Terminal/PowerShell
Keep the backend running, open a NEW terminal window

### 4.2 Navigate to Frontend Directory
```bash
cd frontend
```

### 4.3 Install Node Dependencies
```bash
npm install
```

This will install:
- React (UI library)
- Vite (build tool)
- Tailwind CSS (styling)
- Framer Motion (animations)
- Axios (HTTP requests)
- Lucide Icons (icons)

**Installation time:** 1-3 minutes

### 4.4 Start Development Server
```bash
npm run dev
```

Expected output:
```
  VITE v5.0.0  ready in 450 ms

  ➜  Local:   http://localhost:5173/
```

## Step 5: Access the Application

### ✅ Everything is Running!

Open your browser and visit:

| Component | URL |
|-----------|-----|
| **Application** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |

Try the features:
1. Go to "Vehicle Detection" tab
2. Upload test images from the `traffic/` folder
3. See vehicle detection results
4. View traffic light management on Dashboard

## Step 6: Automated Startup (Optional)

Instead of running commands in two terminals:

### Windows Users:
Double-click `start.bat` in the project root

### macOS/Linux Users:
```bash
chmod +x start.sh
./start.sh
```

Both will:
- ✅ Start backend automatically
- ✅ Start frontend automatically
- ✅ Display URLs for access

## Troubleshooting

### Issue: Python not found
**Solution:**
1. Restart your computer after installing Python
2. Verify installation: `python --version`
3. Use `python3` instead of `python` on macOS/Linux

### Issue: Node.js not found
**Solution:**
1. Restart your computer after installing Node.js
2. Verify: `node --version`
3. Check if terminal was open before installation

### Issue: YOLOv3 model not found
**Solution:**
1. Verify file is in `backend/` directory
2. Check file size: should be ~236 MB
3. Try downloading again if corrupted
4. Use GUI to download: https://pjreddie.com/darknet/yolo/

### Issue: Backend port 8000 already in use
**Solution:**
```bash
# Change port in backend/main.py line 50
# Replace 8000 with 8001:
uvicorn.run(app, host="0.0.0.0", port=8001)

# Then update frontend API URL
# Edit frontend/src/App.jsx
# Replace http://localhost:8000 with http://localhost:8001
```

### Issue: Frontend port 5173 already in use
**Solution:**
```bash
# Specify different port
npm run dev -- --port 5174
```

### Issue: "CUDA not available" message
**Solution:**
This is normal. System will use CPU instead.
To enable GPU:
1. Install NVIDIA CUDA Toolkit
2. Install cuDNN
3. Reinstall opencv-contrib-python[cuda]

### Issue: Out of memory error
**Solution:**
1. Close other applications
2. Restart your computer
3. Use YOLOv3-Tiny instead (smaller model)

### Issue: "Failed to upload file" error
**Solution:**
1. Check backend is running
2. Ensure API is accessible: http://localhost:8000/api/health
3. Verify CORS is enabled (it is by default)
4. Try smaller image files

## Post-Installation

### Recommended Next Steps:

1. **Explore the UI:**
   - Try all 3 tabs (Dashboard, Detection, Live Feed)
   - Upload sample images
   - Start webcam feed

2. **Understand the System:**
   - Read UPGRADE_README.md for overview
   - Review SETUP_GUIDE.md for details

3. **Customize Settings:**
   - Edit detection threshold in `backend/main.py`
   - Change traffic light timing in `frontend/src/components/TrafficLightDashboard.jsx`
   - Modify colors in `frontend/tailwind.config.js`

4. **Test with Your Own Images:**
   - Use images from your local roads
   - Adjust confidence threshold if needed
   - Monitor performance

5. **Deploy (Optional):**
   - Use `docker-compose up` for Docker deployment
   - Follow cloud provider instructions for hosting
   - See SETUP_GUIDE.md for deployment details

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| Disk Space | 2GB | 5GB+ |
| Node.js | 16.x | 18.x LTS |
| Python | 3.8 | 3.10+ |
| Browser | Modern | Chrome/Firefox/Edge |
| GPU | Optional | NVIDIA with CUDA |

## Getting Help

If you encounter issues:

1. **Check troubleshooting section above**
2. **Review console error messages** (usually very helpful)
3. **Verify prerequisites** are installed correctly
4. **Check port conflicts** (8000 and 5173)
5. **Restart everything** (often fixes issues)

## What's Installed?

### Backend
- FastAPI (modern web framework)
- OpenCV (computer vision library)
- NumPy (numerical computing)
- YOLOv3 (object detection model)
- Uvicorn (ASGI server)

### Frontend
- React 18 (UI library)
- Vite 5 (build tool)
- Tailwind CSS (styling framework)
- Framer Motion (animation library)
- Axios (HTTP client)
- Lucide React (icons)

## Quick Command Reference

```bash
# Backend
cd backend
python -m venv venv                # Create virtual environment
source venv/bin/activate          # Activate (macOS/Linux)
pip install -r requirements.txt   # Install packages
python main.py                     # Start server

# Frontend
cd frontend
npm install                        # Install packages
npm run dev                       # Start dev server
npm run build                     # Build for production

# Both
./start.sh                        # Linux/macOS (auto-start both)
start.bat                         # Windows (auto-start both)

# Docker
docker-compose up                 # Start with Docker
```

---

**Installation Complete!** 🎉

You're now ready to use the Traffic Management System.

Visit http://localhost:5173 to get started!
