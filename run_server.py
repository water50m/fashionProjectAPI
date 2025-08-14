#!/usr/bin/env python3
"""
Server runner script with enhanced configuration and error handling
"""

import os
import sys
import json
import argparse
import uvicorn
from pathlib import Path
from prediction_img import YOLOPredictor

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration with defaults"""
    default_config = {
        "API_CONFIG": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": True,
            "log_level": "info"
        }
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return {**default_config, **config}
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return default_config
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return default_config

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'ultralytics',
        'opencv-python',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model_file(config: dict) -> bool:
    """Check if model file exists"""
    model_path = os.path.join(
        config.get("AI_MODEL_PATH", ""),
        config.get("AI_MODEL_NAME", "")
    )
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please check your config.json file and ensure the model file exists.")
        return False
    
    print(f"Model file found: {model_path}")
    return True

def create_directories(config: dict):
    """Create necessary directories"""
    directories = [
        config.get("AI_MODEL_PATH", "./models/"),
        config.get("IMAGE_CONFIG", {}).get("temp_dir", "./temp/")
    ]
    
    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Directory ensured: {directory}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Prediction API Server")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    api_config = config.get("API_CONFIG", {})
    
    # Override with command line arguments
    if args.host:
        api_config["host"] = args.host
    if args.port:
        api_config["port"] = args.port
    if args.reload:
        api_config["reload"] = True
    if args.no_reload:
        api_config["reload"] = False
    
    print("Starting YOLO Prediction API Server...")
    print(f"Config file: {args.config}")
    print(f"Host: {api_config.get('host', '0.0.0.0')}")
    print(f"Port: {api_config.get('port', 8000)}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    create_directories(config)
    
    # Check model file
    if not check_model_file(config):
        print("\nNote: You can download a model with:")
        print("python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        sys.exit(1)
    
    # Start server
    predictor = YOLOPredictor(config_path=args.config)
    try:
        config = predictor.config
        api_config = config.get("API_CONFIG", {})
        
        print("Starting YOLO Prediction API Server...")
        print(f"Host: {api_config.get('host', '0.0.0.0')}")
        print(f"Port: {api_config.get('port', 8000)}")
        print(f"Model: {config.get('AI_MODEL_PATH', '')}/{config.get('AI_MODEL_NAME', '')}")
        
        uvicorn.run(
            app,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000),
            reload=api_config.get("reload", True),
            log_level=api_config.get("log_level", "info")
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()