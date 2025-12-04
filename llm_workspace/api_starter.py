
import sys
sys.path.append('/content/drive/MyDrive/ML_Projects/Customer_Management/llm_workspace')

# Re-import all necessary components
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

# Use the already initialized app from Chapter 10
# (app should already exist in the namespace)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
