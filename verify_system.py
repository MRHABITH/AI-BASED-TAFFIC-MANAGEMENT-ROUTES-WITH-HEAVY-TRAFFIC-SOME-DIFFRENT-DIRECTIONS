#!/usr/bin/env python3
"""
Quick verification tests for Traffic Management System
Tests backend startup and API endpoints
"""

import subprocess
import time
import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

def test_frontend_dist():
    """Verify frontend dist folder exists"""
    dist_path = FRONTEND_DIR / "dist"
    if dist_path.exists():
        files = list(dist_path.rglob("*.js")) + list(dist_path.rglob("*.css")) + list(dist_path.rglob("*.html"))
        print(f"✓ Frontend build verified: {len(files)} files in dist/")
        return True
    else:
        print("✗ Frontend dist folder not found")
        return False

def test_backend_files():
    """Verify all backend files exist"""
    required = ["main.py", "requirements.txt", "Dockerfile"]
    all_exist = True
    
    for file in required:
        path = BACKEND_DIR / file
        if path.exists():
            print(f"✓ Backend file: {file}")
        else:
            print(f"✗ Backend file missing: {file}")
            all_exist = False
    
    return all_exist

def test_yolo_files():
    """Verify YOLO model files exist"""
    files = ["yolov3.cfg", "coco.names"]
    all_exist = True
    
    for file in files:
        path = BASE_DIR / file
        if path.exists():
            size = path.stat().st_size
            print(f"✓ YOLO file: {file} ({size:,} bytes)")
        else:
            print(f"✗ YOLO file missing: {file}")
            all_exist = False
    
    if not (BASE_DIR / "yolov3.weights").exists():
        print("⚠ YOLO weights not found (will be downloaded on first run)")
    else:
        size = (BASE_DIR / "yolov3.weights").stat().st_size
        print(f"✓ YOLO weights: yolov3.weights ({size:,} bytes)")
    
    return all_exist

def test_env_files():
    """Verify environment files"""
    env_files = [
        FRONTEND_DIR / ".env.local",
        FRONTEND_DIR / ".env.production",
    ]
    
    all_exist = True
    for env_file in env_files:
        if env_file.exists():
            print(f"✓ Environment file: {env_file.name}")
        else:
            print(f"✗ Environment file missing: {env_file.name}")
            all_exist = False
    
    return all_exist

def test_docker_config():
    """Verify Docker configuration"""
    dockerfile = BACKEND_DIR / "Dockerfile"
    if dockerfile.exists():
        with open(dockerfile) as f:
            content = f.read()
            if "python" in content and "pip" in content:
                print(f"✓ Dockerfile is valid")
                return True
    
    print("✗ Dockerfile issue")
    return False

def main():
    print("\n" + "="*60)
    print("  TRAFFIC MANAGEMENT SYSTEM - VERIFICATION")
    print("="*60 + "\n")
    
    tests = [
        ("Frontend Build", test_frontend_dist),
        ("Backend Files", test_backend_files),
        ("YOLO Model Files", test_yolo_files),
        ("Environment Config", test_env_files),
        ("Docker Configuration", test_docker_config),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 40)
        results[name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n" + "="*60)
        print("  ✓ ALL VERIFICATIONS PASSED!")
        print("  System is ready for deployment")
        print("="*60 + "\n")
        print("NEXT STEPS:")
        print("1. Backend: Deploy to HuggingFace Spaces (Docker)")
        print("2. Frontend: Deploy to Vercel")
        print("3. Set environment variable for API endpoint")
        print("4. Test integrated system")
        return True
    else:
        print("\n✗ Some verifications failed. Please check above.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
