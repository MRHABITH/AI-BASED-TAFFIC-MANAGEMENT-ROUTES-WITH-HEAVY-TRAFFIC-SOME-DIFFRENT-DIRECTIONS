# Project Structure & File Organization

## 📁 Directory Layout

```
traffic-management-system/
├── frontend/                          # React + Vite Application
│   ├── src/
│   │   ├── components/               # Reusable React components
│   │   │   ├── TrafficLightDashboard.jsx
│   │   │   ├── StatisticsPanel.jsx
│   │   │   ├── VehicleDetectionCard.jsx
│   │   │   ├── VideoFeed.jsx
│   │   │   ├── UIComponents.jsx
│   │   │   └── ErrorBoundary.jsx
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useImageUpload.js
│   │   │   └── useTrafficTimer.js
│   │   ├── services/                 # API communication
│   │   │   └── api.js               # Axios client with interceptors
│   │   ├── utils/                    # Utility functions
│   │   │   ├── validators.js        # Input validation
│   │   │   └── formatters.js        # Data formatting
│   │   ├── assets/                   # Images, icons
│   │   ├── App.jsx                   # Main component
│   │   ├── App.css                   # Global styles
│   │   ├── index.css                 # Tailwind imports
│   │   └── main.jsx                  # Entry point
│   ├── .env.local                    # Development env variables
│   ├── .env.production               # Production env variables
│   ├── .vercelignore                 # Vercel ignore patterns
│   ├── vercel.json                   # Vercel configuration
│   ├── vite.config.js                # Vite build configuration
│   ├── tailwind.config.js            # Tailwind CSS config
│   ├── eslint.config.js              # ESLint configuration
│   ├── package.json                  # Dependencies & scripts
│   └── index.html                    # HTML template
│
├── backend/                           # FastAPI Application
│   ├── main.py                        # FastAPI application & routes
│   ├── requirements.txt               # Python dependencies
│   ├── .env.example                   # Environment variables template
│   ├── Dockerfile                     # Docker image definition
│   ├── yolov3.cfg                     # YOLOv3 configuration
│   ├── yolov3.weights                 # YOLOv3 pre-trained model (236MB)
│   └── coco.names                     # COCO class names
│
├── Deployment Guides/
│   ├── DEPLOYMENT_GUIDE.md            # Master deployment guide
│   ├── BACKEND_DEPLOYMENT.md          # HF Spaces backend deployment
│   ├── FRONTEND_DEPLOYMENT.md         # Vercel frontend deployment
│
├── Documentation/
│   ├── README.md                      # Project overview
│   ├── QUICK_START.md                 # Quick setup guide
│   └── [Other docs...]
│
├── Media/
│   ├── 1.jpg                         # Test images
│   ├── 2.jpg
│   ├── 3.jpg
│   └── demo.jpg
│
└── Configuration/
    ├── .git/                          # Git repository
    ├── .gitignore                     # Git ignore patterns
    ├── docker-compose.yml             # Local Docker setup
    └── start.sh / start.bat           # Quick start scripts
```

## 📊 File Purposes

### Frontend Core
- **main.jsx**: Application entry point
- **App.jsx**: Root component with routing & state
- **index.css**: Global styles & Tailwind imports

### Frontend Components
- **TrafficLightDashboard.jsx**: Traffic light control interface
- **StatisticsPanel.jsx**: Vehicle statistics display
- **VehicleDetectionCard.jsx**: Individual detection results
- **VideoFeed.jsx**: Live stream (if available)
- **UIComponents.jsx**: Reusable components library
- **ErrorBoundary.jsx**: Error handling wrapper

### Frontend Logic
- **useImageUpload.js**: Handle image uploads & validation
- **useTrafficTimer.js**: Traffic light timing logic
- **api.js**: Centralized API communication
- **validators.js**: Input validation utilities
- **formatters.js**: Data formatting functions

### Backend
- **main.py**: FastAPI app with all endpoints
  - `GET /` - Status check
  - `GET /api/health` - Health status
  - `POST /api/detect` - Vehicle detection

### Configuration
- **package.json**: Frontend dependencies & build scripts
- **vite.config.js**: Build optimization
- **vercel.json**: Vercel deployment config
- **requirements.txt**: Backend dependencies
- **Dockerfile**: Backend containerization
- **.env files**: Environment variables per environment

## 🔄 Data Flow

```
User Upload → Frontend → Validation → API Request
    ↓          ↓           ↓           ↓
  Image    FormData      Check      Axios
    ↓        Size/Type     Size      POST
    ↓          ↓           ↓           ↓
           Convert API URL
             ↓
Backend Receive → Process → Detect → Return
    ↓             ↓         ↓       ↓
  Parser    OpenCV       YOLOv3   JSON
Data             ↓         ↓
  ↓              ↓         ↓
Chart.js    Frame        Model    Display
Visual      ↓            ↓         ↓
Stats       NMS         Count    Results
            ↓            ↓
           Filter      Vehicle
           Noise       Types
```

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend UI** | React 19 | Component framework |
| **Styling** | Tailwind CSS v4 | Utility-first CSS |
| **Animation** | Framer Motion | Smooth animations |
| **Icons** | Lucide React | Icon library |
| **Build** | Vite 8 | Fast bundler |
| **HTTP Client** | Axios | API communication |
| **Deployment** | Vercel | Frontend hosting |
| | | |
| **Backend** | FastAPI | Web framework |
| **Server** | Uvicorn | ASGI server |
| **CV** | OpenCV | Image processing |
| **ML** | YOLOv3 | Object detection |
| **Processing** | NumPy | Array processing |
| **Images** | Pillow | Image handling |
| **Containerization** | Docker | Container image |
| **Deployment** | HF Spaces | Backend hosting |

## 📦 Key Dependencies

### Frontend
- react@19.2.4
- vite@8.0.0+
- tailwindcss@4.2.1
- framer-motion@12.36.0
- axios@1.13.6
- lucide-react@0.577.0

### Backend
- fastapi@0.104.1
- uvicorn@0.24.0
- opencv-python@4.8.1.78
- numpy@1.24.3
- pillow@10.0.1

## 🎯 Best Practices

1. **Never commit** sensitive files:
   - `.env` files (use `.env.example`)
   - `yolov3.weights` (use .gitignore)
   - API keys
   - Credentials

2. **Environment isolation**:
   - `.env.local` for development
   - `.env.production` for production
   - HF Spaces secrets for backend

3. **Code organization**:
   - One component per file
   - Keep hooks reusable
   - Utilities should be pure functions
   - Services handle API calls only

4. **Performance**:
   - Lazy load components
   - Optimize images
   - Cache API responses
   - Use code splitting

5. **Security**:
   - Validate all inputs
   - Sanitize data
   - Use HTTPS only
   - Limit CORS origins

## 📝 Naming Conventions

- **Files**: kebab-case for config files, camelCase for code
- **Components**: PascalCase (TrafficLightDashboard.jsx)
- **Functions**: camelCase (processImage)
- **Constants**: UPPER_SNAKE_CASE (VEHICLE_TYPES)
- **CSS Classes**: kebab-case (.traffic-light)

---

**Last Updated**: March 2026
**Maintained By**: Documentation Team
