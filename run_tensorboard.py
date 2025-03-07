#!/usr/bin/env python
"""
Simple script to start TensorBoard for the Energy Market Optimization project.
"""

import os
import argparse
import subprocess
import webbrowser
import time

def main():
    parser = argparse.ArgumentParser(description='Start TensorBoard for Energy Market Optimization')
    parser.add_argument('--logdir', type=str, default='logs/tensorboard',
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--port', type=int, default=6006,
                        help='Port to run TensorBoard on')
    parser.add_argument('--no_browser', action='store_true',
                        help='Do not open browser automatically')
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs(args.logdir, exist_ok=True)
    
    # Check if TensorBoard is installed
    try:
        import tensorboard
        print(f"Using TensorBoard version {tensorboard.__version__}")
    except ImportError:
        print("TensorBoard not found. Installing...")
        subprocess.run(["pip", "install", "tensorboard"])
    
    # Start TensorBoard
    print(f"Starting TensorBoard on port {args.port}...")
    tensorboard_process = subprocess.Popen(
        ["tensorboard", "--logdir", args.logdir, "--port", str(args.port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for TensorBoard to start
    time.sleep(3)
    
    # Open browser
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        print(f"Opening browser at {url}")
        webbrowser.open(url)
    
    print("\nTensorBoard is running. Press Ctrl+C to stop.")
    print(f"View TensorBoard at http://localhost:{args.port}")
    
    try:
        # Keep the script running
        tensorboard_process.wait()
    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")
        tensorboard_process.terminate()
        print("TensorBoard stopped.")

if __name__ == "__main__":
    main() 