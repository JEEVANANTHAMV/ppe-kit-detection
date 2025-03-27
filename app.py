import os
import cv2
import json
import asyncio
from typing import Optional, Dict
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Body,
    HTTPException,
    BackgroundTasks
)
import logging
from helper import (
    check_camera_exists,
    insert_video_record,
    get_tenant_config,
    list_video_records,
    get_video_record,
    update_video_record,
    delete_video_record,
    list_face_records,
    extract_face_embedding,
    trigger_external_event,
    add_face_record,
    update_face_record as update_face_record_helper,
    delete_face_record,
    add_or_update_tenant_config,
    delete_tenant_config,
    update_video_status,
    get_connection,
    format_query,
    list_tenants,
    update_tenant_status,
    ensure_tables_exist
)
from video_processor import start_video_processing
from pydantic import BaseModel, HttpUrl
load_dotenv()
app = FastAPI(title="Safety Violation Detector")
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Make sure database tables are properly set up
try:
    ensure_tables_exist()
    logging.info("Database tables verified successfully")
except Exception as e:
    logging.error(f"Error ensuring database tables exist: {str(e)}")

# Global in-memory dictionary for tracking violation timing only.
violation_timers = {}
violation_timers_lock = asyncio.Lock()

# Store active video processing tasks
active_processes = {}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

######################################################################
# Tenant Config Pydantic Model (with separate columns for each violation)
######################################################################
class TenantConfig(BaseModel):
    similarity_threshold: float
    no_mask_threshold: int
    no_safety_vest_threshold: int
    no_hardhat_threshold: int
    external_trigger_url: Optional[HttpUrl] = None
    is_active: bool = True

class TenantStatusUpdate(BaseModel):
    is_active: bool

######################################################################
# Helper function to get the threshold (in minutes) for a given violation
######################################################################
def get_threshold_for_violation(config: dict, vio: str) -> int:
    if vio == "NO-Mask":
        return config.get("no_mask_threshold", 999)
    elif vio == "NO-Safety Vest":
        return config.get("no_safety_vest_threshold", 999)
    elif vio == "NO-Hardhat":
        return config.get("no_hardhat_threshold", 999)
    return 999

######################################################################
# Helper function to fetch a video record by tenant & camera from the DB
######################################################################
def get_video_record_by_tenant_camera(tenant_id: str, camera_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = """
    SELECT video_id, tenant_id, camera_id, is_live, filename, stream_url, status, size, fps, total_frames, duration, frames_processed, violations_detected 
    FROM videos WHERE tenant_id = ? AND camera_id = ?
    """
    c.execute(format_query(query, db_type), (tenant_id, camera_id))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "video_id": row[0],
            "tenant_id": row[1],
            "camera_id": row[2],
            "is_live": bool(row[3]),
            "filename": row[4],
            "stream_url": row[5],
            "status": row[6],
            "size": row[7],
            "fps": row[8],
            "total_frames": row[9],
            "duration": row[10],
            "frames_processed": row[11],
            "violations_detected": row[12]
        }
    return None

######################################################################
# 1) Video/Stream Registration Endpoint (CHANGED FROM POST TO PUT)
######################################################################
@app.put("/videos")
async def upload_video(
    tenant_id: str = Form(...),
    camera_id: str = Form(...),
    is_live: bool = Form(False),
    stream_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Register or upload a single video/stream for a given tenant & camera.
    For live feeds (is_live=True): supply stream_url only.
    For offline videos (is_live=False): supply a video file only.
    (tenant_id, camera_id) must be unique.
    """
    if check_camera_exists(tenant_id, camera_id):
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' already exists for tenant '{tenant_id}'."
        )

    if is_live:
        if not stream_url or file is not None:
            raise HTTPException(
                status_code=400,
                detail="For a live feed, provide stream_url only (no file)."
            )
        video_id = insert_video_record(
            tenant_id=tenant_id,
            camera_id=camera_id,
            is_live=True,
            stream_url=stream_url
        )
        return {
            "video_id": video_id,
            "tenant_id": tenant_id,
            "camera_id": camera_id,
            "is_live": True,
            "stream_url": stream_url,
            "message": "Live stream registered successfully."
        }
    else:
        if file is None or stream_url is not None:
            raise HTTPException(
                status_code=400,
                detail="For an offline video, provide a file only (no stream_url)."
            )
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        cap = cv2.VideoCapture(file_location)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot open video file: {file.filename}"
            )
        size_bytes = os.path.getsize(file_location)
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fps = original_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        video_id = insert_video_record(
            tenant_id=tenant_id,
            camera_id=camera_id,
            is_live=False,
            filename=file_location,
            size=size_bytes,
            fps=fps,
            total_frames=total_frames,
            duration=duration
        )
        return {
            "video_id": video_id,
            "tenant_id": tenant_id,
            "camera_id": camera_id,
            "is_live": False,
            "filename": file.filename,
            "size": size_bytes,
            "fps": fps,
            "duration": duration,
            "message": "Video uploaded successfully."
        }

######################################################################
# 2) Listing, Status, Stats & DB Status Endpoints
######################################################################
@app.get("/videos")
def list_videos():
    # Now fetch all video records directly from the database
    db_videos = list_video_records()
    return {"videos": db_videos}

@app.get("/status")
def get_status():
    # For status we now query the DB for each video's status
    db_videos = list_video_records()
    status_list = []
    for video in db_videos:
        status_list.append({
            "video_id": video["video_id"],
            "tenant_id": video["tenant_id"],
            "camera_id": video["camera_id"],
            "status": video["status"],
            "frames_processed": video.get("frames_processed", 0),
            "violations_detected": video.get("violations_detected", 0)
        })
    return {"videos_status": status_list}

@app.get("/stats")
def get_stats():
    total_duration = 0
    violation_counts = {}
    processing_videos = []
    db_videos = list_video_records()
    
    for video in db_videos:
        total_duration += video["duration"] if video["duration"] else 0
        violation_counts[video["video_id"]] = video.get("violations_detected", 0)
        
        if video["status"] == "processing":
            processing_videos.append({
                "video_id": video["video_id"],
                "tenant_id": video["tenant_id"],
                "camera_id": video["camera_id"],
                "is_live": video["is_live"],
                "frames_processed": video.get("frames_processed", 0),
                "violations_detected": video.get("violations_detected", 0)
            })
    
    return {
        "total_video_duration": total_duration,
        "violation_counts_per_video": violation_counts,
        "currently_processing": processing_videos,
        "active_processes": len(active_processes)
    }

@app.get("/db-status")
def db_status():
    from helper import get_connection
    conn, db_type = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        status = "connected"
    except Exception as e:
        status = f"error: {str(e)}"
    finally:
        conn.close()
    if db_type == "postgres":
        db_url = os.getenv("DATABASE_URL", "Not set")
        return {"db_type": "postgres", "status": status, "database_url": db_url}
    else:
        from helper import DB_PATH
        return {"db_type": "sqlite", "status": status, "database_path": DB_PATH}

######################################################################
# 3) Processing Videos/Streams in Background
######################################################################
async def monitor_video_process(video_id: str, process, status_queue):
    try:
        while True:
            if not process.is_alive():
                break
                
            try:
                status, message = status_queue.get_nowait()
                if status == "error":
                    logging.error(f"[ERROR] Video {video_id}: {message}")
                    update_video_status(video_id, "error")
                    break
                elif status == "stopped":
                    logging.info(f"[INFO] Video {video_id}: {message}")
                    update_video_status(video_id, "stopped")
                    break
            except:
                pass
                
            await asyncio.sleep(1)
    finally:
        process.terminate()
        process.join()
        if video_id in active_processes:
            del active_processes[video_id]

@app.post("/process")
async def process_videos(background_tasks: BackgroundTasks):
    videos = list_video_records()
    vids_to_run = []
    skipped_videos = []
    
    for video in videos:
        # Skip videos that are not in the correct status
        if video["status"] not in ("uploaded", "registered"):
            continue
            
        config = get_tenant_config(video["tenant_id"])
        
        # Check if tenant config exists and is active
        if not config:
            update_video_status(video["video_id"], "no_config")
            skipped_videos.append({"video_id": video["video_id"], "reason": "No tenant configuration found"})
            continue
            
        if not config.get("is_active", False):
            update_video_status(video["video_id"], "tenant_inactive")
            skipped_videos.append({"video_id": video["video_id"], "reason": "Tenant is inactive"})
            continue
        
        # Check if faces are registered for this tenant
        faces = list_face_records(video["tenant_id"])
        if not faces:
            update_video_status(video["video_id"], "no_faces")
            skipped_videos.append({"video_id": video["video_id"], "reason": "No employee faces registered"})
            continue
            
        vids_to_run.append(video["video_id"])
    
    triggered = []
    for vid in vids_to_run:
        if vid in active_processes:
            continue
            
        video_record = get_video_record(vid)
        config = get_tenant_config(video_record["tenant_id"])
        
        process, status_queue = start_video_processing(
            video_id=vid,
            tenant_id=video_record["tenant_id"],
            camera_id=video_record["camera_id"],
            similarity_threshold=config["similarity_threshold"],
            violation_timers=violation_timers,
            violation_timers_lock=violation_timers_lock
        )
        
        active_processes[vid] = (process, status_queue)
        background_tasks.add_task(monitor_video_process, vid, process, status_queue)
        triggered.append(vid)
    
    if not triggered and not skipped_videos:
        return {"message": "No videos to process for active tenants."}
    
    return {
        "triggered": triggered, 
        "skipped": skipped_videos, 
        "message": f"Triggered processing for {len(triggered)} videos, skipped {len(skipped_videos)} videos."
    }

######################################################################
# 4) Tenant Configuration Endpoints (CHANGED PUT TO POST)
######################################################################
@app.get("/tenants")
def list_tenants_endpoint():
    tenants = list_tenants()
    return {"tenants": tenants}

@app.get("/tenants/{tenant_id}/config")
def get_config(tenant_id: str):
    config = get_tenant_config(tenant_id)
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found.")
    return config

@app.post("/tenants/{tenant_id}/config")
def create_config(tenant_id: str, config: TenantConfig):
    existing = get_tenant_config(tenant_id)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Configuration already exists. Use PUT /tenants/{tenant_id}/config/update to update."
        )
    add_or_update_tenant_config(
        tenant_id,
        config.similarity_threshold,
        config.no_mask_threshold,
        config.no_safety_vest_threshold,
        config.no_hardhat_threshold,
        str(config.external_trigger_url),
        config.is_active
    )
    return {"message": "Configuration created successfully."}

@app.put("/tenants/{tenant_id}/config/update")
def update_config(tenant_id: str, config: TenantConfig):
    existing = get_tenant_config(tenant_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Configuration not found. Use POST /tenants/{tenant_id}/config to create."
        )
    add_or_update_tenant_config(
        tenant_id,
        config.similarity_threshold,
        config.no_mask_threshold,
        config.no_safety_vest_threshold,
        config.no_hardhat_threshold,
        str(config.external_trigger_url),
        config.is_active
    )
    return {"message": "Configuration updated successfully."}

@app.put("/tenants/{tenant_id}/status")
def update_tenant_status_endpoint(tenant_id: str, status: TenantStatusUpdate):
    existing = get_tenant_config(tenant_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Tenant not found.")
    update_tenant_status(tenant_id, status.is_active)
    return {"message": "Tenant status updated successfully."}

@app.delete("/tenants/{tenant_id}/config")
def remove_config(tenant_id: str):
    delete_tenant_config(tenant_id)
    return {"message": "Configuration deleted."}

######################################################################
# 5) Face Management Endpoints (CHANGED TO ACCEPT face_id IN REQUEST)
######################################################################
class FaceCreate(BaseModel):
    tenant_id: str
    camera_id: str
    name: str
    face_id: Optional[str] = None
    metadata: Optional[Dict] = None

class FaceUpdate(BaseModel):
    tenant_id: str
    camera_id: str
    name: Optional[str] = None
    metadata: Optional[Dict] = None

@app.post("/tenants/{tenant_id}/faces")
async def add_face(
    tenant_id: str,
    camera_id: str = Form(...),
    name: str = Form(...),
    face_id: str = Form(...),  # Required face_id from user
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Add a new face for a tenant.
    Validates that the camera exists for the tenant before adding the face.
    Requires face_id to be provided by the user.
    """
    try:
        # Parse metadata if provided
        metadata_dict = json.loads(metadata) if metadata else None
        metadata_json = json.dumps(metadata_dict) if metadata_dict else None

        # Save and process the face image
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        image = cv2.imread(temp_path)
        if image is None:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Invalid image.")
            
        # Extract face embedding
        embedding = extract_face_embedding(image, (0, 0, image.shape[1], image.shape[0]))
        os.remove(temp_path)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected or could not extract embedding.")
        
        # Add face record
        face_id = add_face_record(
            tenant_id=tenant_id,
            camera_id=camera_id,
            face_id=face_id,  # Use provided face_id
            name=name,
            embedding=json.dumps(embedding),
            metadata=metadata_json
        )
        
        return {"face_id": face_id, "message": "Face added successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tenants/{tenant_id}/faces")
def list_faces_endpoint(tenant_id: str):
    faces = list_face_records(tenant_id)
    return {"faces": faces}

@app.put("/tenants/{tenant_id}/faces/{face_id}")
async def update_face(
    tenant_id: str,
    face_id: str,
    camera_id: str = Form(...),
    name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None)
):
    """
    Update an existing face for a tenant.
    Validates that the camera exists for the tenant before updating the face.
    """
    try:
        # Parse metadata if provided
        metadata_dict = json.loads(metadata) if metadata else None
        metadata_json = json.dumps(metadata_dict) if metadata_dict else None
        
        # Process face image if provided
        embedding = None
        if file:
            temp_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            
            image = cv2.imread(temp_path)
            if image is None:
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Invalid image.")
                
            embedding = extract_face_embedding(image, (0, 0, image.shape[1], image.shape[0]))
            os.remove(temp_path)
            
            if embedding is None:
                raise HTTPException(status_code=400, detail="No face detected or could not extract embedding.")
            embedding = json.dumps(embedding)
        
        # Update face record
        update_face_record_helper(
            face_id=face_id,
            tenant_id=tenant_id,
            camera_id=camera_id,
            name=name,
            embedding=embedding,
            metadata=metadata_json
        )
        
        return {"message": "Face updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tenants/{tenant_id}/faces/{face_id}")
def remove_face(tenant_id: str, face_id: str):
    delete_face_record(face_id, tenant_id)
    return {"message": "Face deleted."}

######################################################################
# 6) Video Deletion and Update Endpoints
######################################################################
@app.delete("/videos/{video_id}")
def delete_video(video_id: str):
    success = delete_video_record(video_id)
    if not success:
        raise HTTPException(status_code=404, detail="Video not found.")
    return {"message": f"Video {video_id} deleted successfully."}

@app.put("/videos/{video_id}")
async def update_video(
    video_id: str,
    tenant_id: Optional[str] = Form(None),
    camera_id: Optional[str] = Form(None),
    is_live: Optional[bool] = Form(None),
    stream_url: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    video_record = get_video_record(video_id)
    if not video_record:
        raise HTTPException(status_code=404, detail="Video not found.")
        
    # Check if trying to upload file for live stream
    if file is not None and (video_record["is_live"] or (is_live is not None and is_live)):
        raise HTTPException(
            status_code=400,
            detail="Cannot upload file for live stream. Use stream_url instead."
        )
        
    update_fields = {}
    if tenant_id is not None:
        update_fields["tenant_id"] = tenant_id
    if camera_id is not None:
        update_fields["camera_id"] = camera_id
    if is_live is not None:
        update_fields["is_live"] = 1 if is_live else 0
    if stream_url is not None:
        update_fields["stream_url"] = stream_url
    if status is not None:
        update_fields["status"] = status

    if file and not video_record["is_live"]:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        cap = cv2.VideoCapture(file_location)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Cannot open video file: {file.filename}")
        try:
            size_bytes = os.path.getsize(file_location)
            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            fps = original_fps
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            update_fields["filename"] = file_location
            update_fields["size"] = size_bytes
            update_fields["fps"] = fps
            update_fields["total_frames"] = total_frames
            update_fields["duration"] = duration
        finally:
            cap.release()

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    update_video_record(video_id, update_fields)
    return {"message": f"Video {video_id} updated successfully."}

######################################################################
# 7) External Trigger Testing Endpoint (IMPROVED ERROR HANDLING)
######################################################################
@app.post("/trigger-event")
async def trigger_event(payload: dict = Body(...)):
    tenant_id = payload.get("tenant_id")
    camera_id = payload.get("camera_id")
    event_url = payload.get("event_url")
    if not tenant_id or not camera_id or not event_url:
        raise HTTPException(status_code=400, detail="tenant_id, camera_id, and event_url are required.")
    
    try:
        status_code, response_text = await trigger_external_event(event_url, payload)
        return {"status_code": status_code, "response_text": response_text}
    except Exception as e:
        return {"status_code": 500, "response_text": f"Error triggering event: {str(e)}"}

######################################################################
# 8) Tenants & Live URL Endpoints
######################################################################
@app.put("/tenants/{tenant_id}/cameras/{camera_id}/live-url")
def update_live_url(tenant_id: str, camera_id: str, stream_url: str = Body(...)):
    # Update the live stream URL for a specific tenant and camera.
    video = get_video_record_by_tenant_camera(tenant_id, camera_id)
    if not video:
        raise HTTPException(status_code=404, detail="Camera not found for the given tenant.")
    if not video["is_live"]:
        raise HTTPException(status_code=400, detail="The specified camera is not a live feed.")
    update_video_record(video["video_id"], {"stream_url": stream_url})
    return {"message": "Live URL updated successfully."}
