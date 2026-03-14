#!/usr/bin/env python3
"""
Comprehensive test suite for Traffic Management System
Tests backend API, frontend build, and project configuration
"""

import os
import sys
import json
import requests
import subprocess
import time
from pathlib import Path

# Define base paths
BASE_DIR = Path(__file__).parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(msg):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    """Print error message"""
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def test_project_structure():
    """Test if all required files exist"""
    print_section("1. PROJECT STRUCTURE TEST")
    
    required_files = {
        "backend/main.py": "Backend API",
        "backend/requirements.txt": "Backend dependencies",
        "backend/Dockerfile": "Docker configuration",
        "frontend/package.json": "Frontend dependencies",
        "frontend/vite.config.js": "Vite configuration",
        "frontend/src/App.jsx": "Main React component",
        "yolov3.cfg": "YOLO config",
        "coco.names": "COCO classes",
        "README.md": "Project documentation",
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        full_path = BASE_DIR / file_path
        if full_path.exists():
            print_success(f"{description}: {file_path}")
        else:
            print_error(f"{description}: {file_path} (MISSING)")
            all_exist = False
    
    return all_exist

def test_backend_syntax():
    """Test backend Python syntax"""
    print_section("2. BACKEND SYNTAX TEST")
    
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", "main.py"],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print_success("Backend Python syntax is valid")
            return True
        else:
            print_error(f"Syntax error: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Failed to check syntax: {e}")
        return False

def test_backend_deps():
    """Test backend dependencies"""
    print_section("3. BACKEND DEPENDENCIES TEST")
    
    req_file = BACKEND_DIR / "requirements.txt"
    if not req_file.exists():
        print_warning("requirements.txt not found")
        return False
    
    with open(req_file) as f:
        deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Found {len(deps)} dependencies:")
    critical_deps = ["fastapi", "opencv-python", "numpy", "uvicorn"]
    
    missing = []
    for dep_name in critical_deps:
        found = any(dep_name.lower() in dep.lower() for dep in deps)
        if found:
            print_success(f"{dep_name} found in requirements.txt")
        else:
            print_warning(f"{dep_name} NOT found in requirements.txt")
            missing.append(dep_name)
    
    return len(missing) == 0

def test_frontend_build():
    """Test if frontend builds successfully"""
    print_section("4. FRONTEND BUILD TEST")
    
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=FRONTEND_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            if (FRONTEND_DIR / "dist").exists():
                # Count built files
                dist_files = list((FRONTEND_DIR / "dist").rglob("*"))
                file_count = len([f for f in dist_files if f.is_file()])
                print_success(f"Frontend build successful ({file_count} files generated)")
                return True
            else:
                print_error("Frontend build completed but dist/ folder not found")
                return False
        else:
            print_error(f"Frontend build failed: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Frontend build timeout (>60s)")
        return False
    except Exception as e:
        print_error(f"Frontend build error: {e}")
        return False

def test_env_config():
    """Test environment configuration"""
    print_section("5. ENVIRONMENT CONFIGURATION TEST")
    
    env_files = {
        "frontend/.env.local": "Development API endpoint",
        "frontend/.env.production": "Production API endpoint",
        "backend/.env": "Backend environment (optional)",
    }
    
    env_count = 0
    for env_file, description in env_files.items():
        env_path = BASE_DIR / env_file
        if env_path.exists():
            print_success(f"{description}: {env_file}")
            env_count += 1
        else:
            if "optional" in description:
                print_warning(f"{description}: {env_file} (optional)")
            else:
                print_warning(f"{description}: {env_file}")
    
    return env_count >= 1

def test_docker_config():
    """Test Docker configuration"""
    print_section("6. DOCKER CONFIGURATION TEST")
    
    docker_file = BACKEND_DIR / "Dockerfile"
    if docker_file.exists():
        with open(docker_file) as f:
            content = f.read()
        
        checks = {
            "FROM": "Base image",
            "WORKDIR": "Working directory",
            "COPY": "File copying",
            "CMD": "Runtime command",
            "python": "Python image",
        }
        
        all_found = True
        for check, desc in checks.items():
            if check in content:
                print_success(f"Dockerfile contains {desc}")
            else:
                print_warning(f"Dockerfile missing {desc}")
                all_found = False
        
        return all_found
    else:
        print_warning("Dockerfile not found")
        return False

def test_api_endpoints():
    """Test API endpoints are properly defined"""
    print_section("7. API ENDPOINTS TEST")
    
    main_file = BACKEND_DIR / "main.py"
    with open(main_file) as f:
        content = f.read()
    
    endpoints = {
        '@app.get("/")': "Root endpoint",
        '@app.post("/api/detect")': "Vehicle detection endpoint",
        '@app.get("/health")': "Health check endpoint",
    }
    
    all_found = True
    for endpoint, desc in endpoints.items():
        if endpoint in content:
            print_success(f"{desc}: {endpoint}")
        else:
            print_error(f"{desc} NOT FOUND: {endpoint}")
            all_found = False
    
    return all_found

def test_documentation():
    """Test project documentation"""
    print_section("8. DOCUMENTATION TEST")
    
    doc_files = [
        "README.md",
        "DEPLOYMENT_GUIDE.md",
        "BACKEND_DEPLOYMENT.md",
        "FRONTEND_DEPLOYMENT.md",
        "PROJECT_STRUCTURE.md",
    ]
    
    doc_count = 0
    for doc in doc_files:
        doc_path = BASE_DIR / doc
        if doc_path.exists():
            print_success(f"Documentation found: {doc}")
            doc_count += 1
        else:
            print_warning(f"Documentation missing: {doc}")
    
    return doc_count >= 2

def test_config_validity():
    """Test configuration file validity"""
    print_section("9. CONFIGURATION VALIDITY TEST")
    
    # Test vite.config.js
    vite_config = FRONTEND_DIR / "vite.config.js"
    if vite_config.exists():
        try:
            with open(vite_config) as f:
                content = f.read()
            # Basic checks
            if "export default" in content and "defineConfig" in content:
                print_success("Vite configuration is valid")
            else:
                print_warning("Vite configuration may be incomplete")
        except Exception as e:
            print_warning(f"Vite config check failed: {e}")
    
    # Test package.json
    pkg_json = FRONTEND_DIR / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json) as f:
                pkg = json.load(f)
            
            scripts = pkg.get("scripts", {})
            required_scripts = ["dev", "build", "preview"]
            
            for script in required_scripts:
                if script in scripts:
                    print_success(f"npm script found: {script}")
                else:
                    print_warning(f"npm script missing: {script}")
        except json.JSONDecodeError:
            print_error("package.json is not valid JSON")
            return False
    
    return True

def generate_test_report():
    """Generate comprehensive test report"""
    print_section("RUNNING COMPREHENSIVE TEST SUITE")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Backend Syntax", test_backend_syntax),
        ("Backend Dependencies", test_backend_deps),
        ("Frontend Build", test_frontend_build),
        ("Environment Config", test_env_config),
        ("Docker Config", test_docker_config),
        ("API Endpoints", test_api_endpoints),
        ("Documentation", test_documentation),
        ("Config Validity", test_config_validity),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}\n")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{status}{Colors.END} - {test_name}")
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    
    if passed == total:
        print_success("ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        failed = total - passed
        print_warning(f"{failed} test(s) failed. Please review above.")
        return False

if __name__ == "__main__":
    success = generate_test_report()
    sys.exit(0 if success else 1)
