# Quick Reference & Troubleshooting

## 🚀 Quick Commands

### Start Everything at Once
```bash
# Windows
start.bat

# macOS/Linux  
./start.sh
```

### Manual Start (2 Terminals)
```bash
# Terminal 1 - Backend
cd backend
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### URLs to Access
- **App:** http://localhost:5173
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 🔍 Quick Troubleshooting

### ❌ "Cannot find module" or Dependencies missing

**Solution:**
```bash
# Frontend
cd frontend
npm install
npm install axios framer-motion lucide-react

# Backend
cd backend
pip install -r requirements.txt
```

---

### ❌ Port Already in Use (8000 or 5173)

**Check what's using the port:**
```bash
# Windows (PowerShell)
Get-NetTCPConnection -LocalPort 8000

# macOS/Linux
lsof -i :8000
```

**Kill the process or use different port:**
```bash
# Backend - Edit backend/main.py last line
python main.py --port 8001

# Frontend  
npm run dev -- --port 5174
```

---

### ❌ Backend error: "Module 'cv2' not found"

**Solution:**
```bash
cd backend
pip install --upgrade opencv-python opencv-contrib-python
```

---

### ❌ YOLOv3 weights file not found

**Verify file exists:**
```bash
cd backend
ls -la yolov3.weights  # macOS/Linux
dir yolov3.weights     # Windows

# Should show ~236 MB file
```

**Download if missing:**
```bash
# Windows
powershell -Command "Invoke-WebRequest -Uri 'https://pjreddie.com/media/files/yolov3.weights' -OutFile 'yolov3.weights'"

# macOS/Linux
wget https://pjreddie.com/media/files/yolov3.weights
```

---

### ❌ Backend connection failed in browser

**Test API manually:**
```bash
# Windows PowerShell
Invoke-WebRequest http://localhost:8000/api/health

# macOS/Linux
curl http://localhost:8000/api/health

# Should return:
# {"status":"healthy","model":"YOLOv3"}
```

---

### ❌ Camera permission denied

**Browser Settings:**
1. Firefox: Preferences → Privacy → Permissions → Camera
2. Chrome: Settings → Privacy → Site Settings → Camera
3. Edge: Settings → Privacy → Site Permissions → Camera
4. Safari: Preferences → Privacy → Camera

---

### ❌ Images won't upload

**Check:**
1. Backend is running (http://localhost:8000/api/health returns OK)
2. Image file is under 5MB
3. Image format is JPEG or PNG
4. Firebase/VPN not blocking requests

**Try in console:**
```javascript
// In browser DevTools → Console
fetch('http://localhost:8000/api/health')
  .then(r => r.json())
  .then(console.log)
  .catch(console.error)
```

---

### ⚠️ "CUDA not available" warning

**This is normal.** Your system will use CPU instead.

To enable NVIDIA GPU:
1. Install NVIDIA CUDA Toolkit
2. Install cuDNN
3. Rebuild OpenCV with CUDA support

**For now, CPU works fine (just slower).**

---

### ❌ "Out of memory" error

**Solutions:**
1. Close other applications
2. Restart your computer
3. Use smaller images
4. Update to YOLOv3-Tiny (faster/smaller)

---

## 📊 Performance Checklist

| Check | How to Test | Expected |
|-------|-------------|----------|
| Frontend loading | Open http://localhost:5173 | <1 second |
| API response | Upload image | <1 second |
| Detection speed | Check console logs | <200ms |
| Frame rate | Open DevTools F12 → Perf tab | 60 FPS |
| Memory | Task Manager/Activity Monitor | <500MB total |

---

## 🐛 Debug Mode

### Enable Debug Logging

**Frontend - Edit src/App.jsx:**
```javascript
const handleImageUpload = async (e) => {
  console.log('Starting upload...', files);
  console.log('API Response:', response.data);
}
```

**Backend - Edit backend/main.py:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Processing image...")
```

**Browser - Open DevTools:**
- Windows/Linux: F12 or Ctrl+Shift+I
- macOS: Cmd+Option+I

---

## 🔐 Security Notes

⚠️ **For Development Only:**
- CORS allows all origins (change in production)
- No authentication required
- No input validation for production

**For Production:**
1. Restrict CORS to your domain
2. Add authentication
3. Validate file uploads
4. Use HTTPS
5. Add rate limiting

---

## 📱 Testing on Mobile

### From Same Computer
```bash
# Get your IP address
# Windows: ipconfig (look for IPv4 Address)
# macOS/Linux: ifconfig (look for inet)

# Then visit:
http://YOUR_IP:5173
```

### From Different Computer
1. Ensure backend on same network
2. Use your computer's IP (not localhost)
3. Make sure firewall allows port 5173

---

## 🚢 Build for Production

### Frontend
```bash
cd frontend
npm run build
# Creates 'dist' folder
# Upload to Vercel, Netlify, AWS S3, etc.
```

### Backend
```bash
cd backend
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
# Or use Docker
docker build -t traffic-api .
docker run -p 8000:8000 traffic-api
```

---

## 📚 File Locations Reference

```
Project Root/
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Main component
│   │   ├── index.css            ← Global styles
│   │   └── components/
│   │       ├── VehicleDetectionCard.jsx
│   │       ├── TrafficLightDashboard.jsx
│   │       ├── StatisticsPanel.jsx
│   │       └── VideoFeed.jsx
│   ├── vite.config.js           ← Build config
│   ├── tailwind.config.js       ← Styling config
│   └── package.json
│
├── backend/
│   ├── main.py                  ← API server
│   ├── requirements.txt
│   ├── yolov3.cfg
│   ├── coco.names
│   └── yolov3.weights          ← Download separately
│
├── INSTALLATION.md              ← Step-by-step guide
├── SETUP_GUIDE.md              ← Full documentation
├── UPGRADE_README.md           ← Quick start
└── MIGRATION_SUMMARY.md        ← What changed
```

---

## 🔧 Common Edits

### Change Traffic Light Timing
**File:** `frontend/src/components/TrafficLightDashboard.jsx`
```javascript
const totalGreenTime = 30;  // Change to your value
```

### Change API Endpoint
**File:** `frontend/src/App.jsx`
```javascript
const response = await axios.post('http://localhost:8000/api/detect', ...);
// Change 8000 to your API port
```

### Change Detection Confidence
**File:** `backend/main.py`
```python
if confidence > 0.5:  # Change 0.5 to 0.3-0.9
```

### Change UI Colors
**File:** `frontend/tailwind.config.js`
```javascript
colors: {
  primary: '#0f172a',    // Dark blue
  secondary: '#1e293b',  // Lighter blue
  accent: '#3b82f6',     // Bright blue
}
```

---

## 🎯 Success Checklist

After setup, verify everything:

- [ ] Backend running at http://localhost:8000
- [ ] API health check works
- [ ] Frontend loads at http://localhost:5173
- [ ] Can upload images without errors
- [ ] Vehicle detection works
- [ ] Traffic light dashboard appears
- [ ] Animations are smooth
- [ ] No errors in browser console
- [ ] No errors in backend terminal
- [ ] Responsive design on mobile browser

---

## 📞 Need Help?

**Check These Files In Order:**
1. `INSTALLATION.md` - Setup problems
2. `SETUP_GUIDE.md` - Feature questions
3. `Console errors` - Copy full error message
4. Browser DevTools → Console tab
5. Backend terminal output

**Common Keywords to Search For:**
- Your error message
- Port number error
- Module not found
- Connection refused
- Permission denied

---

## ⏱️ Typical Issues & Time to Fix

| Issue | Time | Difficulty |
|-------|------|-----------|
| Dependencies missing | 2min | ⭐ Easy |
| Port conflict | 3min | ⭐ Easy |
| YOLO weights missing | 5min | ⭐ Easy |
| Backend not responding | 5min | ⭐⭐ Medium |
| CUDA/GPU issues | 30min | ⭐⭐⭐ Hard |
| Network/firewall | 10min | ⭐⭐ Medium |

---

## 💡 Pro Tips

1. **Keep terminal windows organized**
   - One for backend, one for frontend
   - Keep them visible side-by-side

2. **Use VS Code integrated terminal**
   - Split terminals for front/back
   - Better for debugging

3. **Monitor logs**
   - Backend logs show API calls
   - Frontend console shows errors

4. **Test with different images**
   - Use various road images
   - Check detection accuracy

5. **Use browser DevTools**
   - Network tab for API calls
   - Console for JavaScript errors
   - Performance tab for FPS

---

**Last Updated:** March 13, 2026
**Version:** 2.0 React + Vite
**Status:** Verified Working ✓
