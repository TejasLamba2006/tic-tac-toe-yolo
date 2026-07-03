import subprocess
import sys
import shutil
import os

def has_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None

def main():
    print("Checking for NVIDIA GPU...")
    gpu_detected = has_nvidia_gpu()
    
    if gpu_detected:
        print("NVIDIA GPU detected. Using CUDA 13.2 index...")
        index_url = "https://download.pytorch.org/whl/cu132"
    else:
        print("No NVIDIA GPU detected. Using CPU index...")
        index_url = "https://download.pytorch.org/whl/cpu"
    
    if shutil.which("uv") is None:
        print("Error: 'uv' is not installed or not in PATH.")
        sys.exit(1)
        
    # Check if a virtual environment exists, if not, create one
    if not os.path.exists(".venv") and not os.environ.get("VIRTUAL_ENV"):
        print("Creating virtual environment with uv...")
        subprocess.run(["uv", "venv"], check=True)
        print("Virtual environment created.")
        
    cmd = [
        "uv", "pip", "install", "-e", ".[training]",
        "--index-url", index_url,
        "--extra-index-url", "https://pypi.org/simple"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # We use shell=True on Windows if running a command that might need it, 
    # but with list of args, shell=False is generally fine. 
    # We pass the environment so that uv picks up the .venv if we just created it.
    try:
        subprocess.run(cmd, check=True)
        print("\nSuccessfully installed training dependencies!")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
