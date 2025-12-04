"""
================================================================================
INITIALIZATION SCRIPT
Starts: Ollama + FastAPI + Streamlit
================================================================================
"""

import subprocess
import time
import sys
from pathlib import Path
import requests

print("="*80)
print(" "*15 + "🚀 INITIALIZING FULL SYSTEM")
print("="*80)
print()

# ============================================================================
# STEP 1: Start Ollama
# ============================================================================

print("📄 STEP 1: Starting Ollama...")
print("-" * 40)

try:
    # Check if already running
    result = subprocess.run(['pgrep', 'ollama'], capture_output=True)
    
    if result.returncode != 0:
        # Start Ollama
        print("  Starting Ollama server...")
        subprocess.Popen(
            ['nohup', 'ollama', 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(10)
        print("  ✅ Ollama started")
    else:
        print("  ✅ Ollama already running")

except Exception as e:
    print(f"  ❌ Error starting Ollama: {e}")
    sys.exit(1)

print()

# ============================================================================
# STEP 2: Start FastAPI
# ============================================================================

print("📄 STEP 2: Starting FastAPI...")
print("-" * 40)

# Detect project root
def get_project_root():
    try:
        import google.colab
        return Path("/content/drive/MyDrive/ML_Projects/Customer_Management")
    except:
        return Path(__file__).parent.parent

PROJECT_ROOT = get_project_root()
api_script = PROJECT_ROOT / "llm_workspace" / "api.py"

if not api_script.exists():
    print(f"  ❌ API script not found: {api_script}")
    print("  Expected location: llm_workspace/api.py")
    sys.exit(1)

try:
    # Check if already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("  ✅ FastAPI already running")
        else:
            raise Exception("Not running")
    except:
        # Start FastAPI
        print("  Starting FastAPI server...")
        subprocess.Popen(
            [
                sys.executable, '-m', 'uvicorn',
                'api:app',
                '--host', '0.0.0.0',
                '--port', '8000'
            ],
            cwd=str(api_script.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for API to be ready
        print("  Waiting for API to be ready...")
        for i in range(30):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("  ✅ FastAPI started and healthy")
                    break
            except:
                continue
        else:
            print("  ⚠️  API may not be fully ready yet")

except Exception as e:
    print(f"  ❌ Error starting FastAPI: {e}")
    sys.exit(1)

print()

# ============================================================================
# STEP 3: Start Streamlit
# ============================================================================

print("📄 STEP 3: Starting Streamlit...")
print("-" * 40)

ui_main = PROJECT_ROOT / "ui_files" / "main.py"

if not ui_main.exists():
    print(f"  ❌ Streamlit main.py not found: {ui_main}")
    sys.exit(1)

try:
    print("  Starting Streamlit server...")
    subprocess.Popen(
        [
            'streamlit', 'run',
            str(ui_main),
            '--server.port', '8501',
            '--server.headless', 'true'
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(5)
    print("  ✅ Streamlit started")

except Exception as e:
    print(f"  ❌ Error starting Streamlit: {e}")
    sys.exit(1)

print()

# ============================================================================
# STEP 4: Setup Ngrok (if in Colab)
# ============================================================================

print("📄 STEP 4: Setting up Ngrok...")
print("-" * 40)

try:
    from pyngrok import ngrok
    
    # Kill existing tunnels
    ngrok.kill()
    
    # Create new tunnel
    public_url = ngrok.connect(8501)
    
    print(f"  ✅ Ngrok tunnel created")
    print()
    print("="*80)
    print()
    print("🎉 SYSTEM READY!")
    print()
    print(f"🌐 Access UI at: {public_url}")
    print()
    print("📊 Services running:")
    print("   - Ollama: localhost:11434")
    print("   - FastAPI: http://localhost:8000")
    print("   - Streamlit: http://localhost:8501")
    print(f"   - Public URL: {public_url}")
    print()
    print("="*80)

except ImportError:
    print("  ⚠️  Ngrok not available (not in Colab)")
    print()
    print("="*80)
    print()
    print("🎉 SYSTEM READY (Local Mode)!")
    print()
    print("📊 Services running:")
    print("   - Ollama: localhost:11434")
    print("   - FastAPI: http://localhost:8000")
    print("   - Streamlit: http://localhost:8501")
    print()
    print("="*80)

except Exception as e:
    print(f"  ⚠️  Ngrok error: {e}")
    print("  You can still access locally at http://localhost:8501")
