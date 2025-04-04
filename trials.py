from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import logging
import json

app = FastAPI()

class ViolationPayload(BaseModel):
    id: str
    tenant_id: str
    camera_id: str
    violation_timestamp: float
    face_id: str
    violation_type: str
    violation_image_path: str
    details: str

@app.post("/trigger")
async def handle_violation(payload: ViolationPayload):
    try:
        # Log the incoming violation
        logging.info(f"Received violation: {payload.violation_type} for face {payload.face_id}")
        
        # Parse the details JSON string
        details = json.loads(payload.details)
       
        return {
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error processing violation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process violation: {str(e)}"
        )

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)