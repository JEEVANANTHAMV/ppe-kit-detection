import os
import cv2
import time
import json
import asyncio
import datetime
import torch
from typing import List, Union, Optional

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
    init_db,
    check_camera_exists,
    insert_video_record,
    get_tenant_config,
    list_video_records,
    get_video_record,
    update_video_record,
    delete_video_record,
    list_face_records,
    compare_embeddings,
    detect_objects,
    check_violations,
    save_annotated_plot,
    save_violation_to_db,
    extract_face_embedding,
    trigger_external_event,
    add_face_record,
    update_face_record as update_face_record_helper,
    delete_face_record,
    add_or_update_tenant_config,
    delete_tenant_config,
    DB_PATH
)
from pydantic import BaseModel, HttpUrl
from typing import Dict

app = FastAPI(title="Safety Violation Detector")
init_db()
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Global in-memory state for runtime tracking
videos_info = {}  # video_id -> metadata dict
violation_timers = {}

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

######################################################################
# Helper function to get the threshold (in minutes) for a given violation
######################################################################
def get_threshold_for_violation(config: dict, vio: str) -> int:
    if vio == "No-Mask":
        return config.get("no_mask_threshold", 9999)
    elif vio == "No-Safety-Vest":
        return config.get("no_safety_vest_threshold", 9999)
    elif vio == "No-Hardhat":
        return config.get("no_hardhat_threshold", 9999)
    return 9999

######################################################################
# 1) Video/Stream Registration Endpoint
######################################################################
@app.post("/videos")
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
        videos_info[video_id] = {
            "video_id": video_id,
            "tenant_id": tenant_id,
            "camera_id": camera_id,
            "is_live": True,
            "stream_url": stream_url,
            "filename": None,
            "size": 0,
            "fps": 0,
            "total_frames": 0,
            "duration": 0,
            "status": "registered",
            "frames_processed": 0,
            "violations_detected": 0
        }
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
        videos_info[video_id] = {
            "video_id": video_id,
            "tenant_id": tenant_id,
            "camera_id": camera_id,
            "is_live": False,
            "stream_url": None,
            "filename": file_location,
            "size": size_bytes,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "status": "uploaded",
            "frames_processed": 0,
            "violations_detected": 0
        }
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
    # Retrieve persistent video records from the DB
    db_videos = list_video_records()
    # Merge with runtime state if available
    for video in db_videos:
        vid_id = video["video_id"]
        if vid_id in videos_info:
            runtime_info = videos_info[vid_id]
            video["frames_processed"] = runtime_info.get("frames_processed", 0)
            video["violations_detected"] = runtime_info.get("violations_detected", 0)
        else:
            video["frames_processed"] = 0
            video["violations_detected"] = 0
    return {"videos": db_videos}

@app.get("/status")
def get_status():
    status_list = []
    for vid_id, meta in videos_info.items():
        status_list.append({
            "video_id": vid_id,
            "tenant_id": meta["tenant_id"],
            "camera_id": meta["camera_id"],
            "status": meta["status"],
            "frames_processed": meta["frames_processed"],
            "violations_detected": meta["violations_detected"]
        })
    return {"videos_status": status_list}

@app.get("/stats")
def get_stats():
    total_duration = 0
    violation_counts = {}
    db_videos = list_video_records()
    for video in db_videos:
        total_duration += video["duration"] if video["duration"] else 0
        vid_id = video["video_id"]
        if vid_id in videos_info:
            violation_counts[vid_id] = videos_info[vid_id].get("violations_detected", 0)
        else:
            violation_counts[vid_id] = 0
    total_processed_duration = sum(v.get("frames_processed", 0) / (v.get("fps", 30) or 30) for v in videos_info.values())
    return {
        "total_video_duration": total_duration,
        "total_processed_duration": total_processed_duration,
        "violation_counts_per_video": violation_counts
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
        return {"db_type": "sqlite", "status": status, "database_path": DB_PATH}

######################################################################
# 3) Processing Videos/Streams in Background
######################################################################
async def process_video_stream(video_id: int):
    if video_id not in videos_info:
        return
    meta = videos_info[video_id]
    tenant_id = meta["tenant_id"]
    camera_id = meta["camera_id"]
    config = get_tenant_config(tenant_id)
    if not config:
        logging.error(f"[ERROR] No tenant configuration for tenant {tenant_id}")
        meta["status"] = "error"
        return

    similarity_threshold = config["similarity_threshold"]
    external_trigger_url = config["external_trigger_url"]

    cap_source = meta["stream_url"] if meta["is_live"] else meta["filename"]
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logging.error(f"[ERROR] Cannot open source for video_id {video_id}")
        meta["status"] = "error"
        return

    meta["status"] = "processing"
    use_gpu = torch.cuda.is_available()
    target_fps = 15 if use_gpu else 3
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip_interval = max(1, int(original_fps // target_fps))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue

        boxes, results, annotated_img, colors, class_confidences, violation_faces = await asyncio.to_thread(
            detect_objects, frame
        )
        violations_found = check_violations(class_confidences, threshold=0.5)
        if violations_found:
            annotated_path = save_annotated_plot(
                original_bgr_image=frame,
                boxes=boxes,
                class_names=results.names,
                annotated_image_rgb=annotated_img,
                colors=colors,
                confidence_threshold=0.5,
                video_source=cap_source
            )
            current_time = time.time()
            for person_id, (x1, y1, x2, y2) in violation_faces:
                embedding = extract_face_embedding(frame, (x1, y1, x2, y2))
                if embedding is None:
                    for vio in violations_found:
                        key = (tenant_id, camera_id, person_id, vio)
                        violation_timers.pop(key, None)
                    continue

                matched_face_id = None
                all_faces = list_face_records(tenant_id)
                for face_rec in all_faces:
                    if face_rec["camera_id"] != camera_id:
                        continue
                    if compare_embeddings(embedding, face_rec["embedding"], similarity_threshold):
                        matched_face_id = face_rec["face_id"]
                        break
                if not matched_face_id:
                    continue

                for vio in violations_found:
                    key = (tenant_id, camera_id, matched_face_id, vio)
                    threshold_minutes = get_threshold_for_violation(config, vio)
                    threshold_seconds = threshold_minutes * 60
                    if key not in violation_timers:
                        violation_timers[key] = current_time
                    else:
                        elapsed = current_time - violation_timers[key]
                        if elapsed >= threshold_seconds:
                            violation_record = {
                                "tenant_id": tenant_id,
                                "camera_id": camera_id,
                                "violation_timestamp": current_time,
                                "face_id": matched_face_id,
                                "violation_type": vio,
                                "violation_image_path": annotated_path,
                                "details": json.dumps({"person_id": person_id, "elapsed_seconds": elapsed})
                            }
                            save_violation_to_db(**violation_record)
                            violation_timers[key] = current_time
                            meta["violations_detected"] += 1
                            if external_trigger_url:
                                payload = violation_record.copy()
                                asyncio.create_task(trigger_external_event(external_trigger_url, payload))
                for key in list(violation_timers.keys()):
                    if key[0] == tenant_id and key[1] == camera_id and key[2] == matched_face_id and key[3] not in violations_found:
                        violation_timers.pop(key, None)

        meta["frames_processed"] += frame_skip_interval
        await asyncio.sleep(0)
    cap.release()
    meta["status"] = "done"
    logging.info(f"[INFO] Finished processing video_id={video_id} (tenant={tenant_id}, camera={camera_id}).")

@app.post("/process")
async def process_videos(
    video_ids: Union[List[int], str] = Body(...),
    background_tasks: BackgroundTasks = None
):
    if isinstance(video_ids, str) and video_ids.strip() == "*":
        vids_to_run = list(videos_info.keys())
    elif isinstance(video_ids, list):
        vids_to_run = video_ids
    else:
        raise HTTPException(status_code=400, detail="Invalid input. Must be '*' or list of video IDs.")
    if not vids_to_run:
        return {"message": "No videos to process."}
    triggered = []
    for vid in vids_to_run:
        if vid in videos_info:
            background_tasks.add_task(process_video_stream, vid)
            triggered.append(vid)
    return {"message": f"Triggered processing for videos: {triggered}"}

######################################################################
# 4) Tenant Configuration Endpoints
######################################################################
@app.get("/tenants/{tenant_id}/config")
def get_config(tenant_id: str):
    config = get_tenant_config(tenant_id)
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found.")
    return config

@app.post("/tenants/{tenant_id}/config")
def create_config(tenant_id: str, config: TenantConfig):
    add_or_update_tenant_config(
        tenant_id,
        config.similarity_threshold,
        config.no_mask_threshold,
        config.no_safety_vest_threshold,
        config.no_hardhat_threshold,
        config.external_trigger_url
    )
    return {"message": "Configuration created/updated."}

@app.put("/tenants/{tenant_id}/config")
def update_config(tenant_id: str, config: TenantConfig):
    add_or_update_tenant_config(
        tenant_id,
        config.similarity_threshold,
        config.no_mask_threshold,
        config.no_safety_vest_threshold,
        config.no_hardhat_threshold,
        config.external_trigger_url
    )
    return {"message": "Configuration updated."}

@app.delete("/tenants/{tenant_id}/config")
def remove_config(tenant_id: str):
    delete_tenant_config(tenant_id)
    return {"message": "Configuration deleted."}

######################################################################
# 5) Face Management Endpoints
######################################################################
@app.post("/tenants/{tenant_id}/faces")
async def add_face(
    tenant_id: str,
    camera_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    metadata: str = Form(...),
):
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON.")

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
    face_id = add_face_record(tenant_id, camera_id, name, embedding, metadata_dict)
    return {"message": "Face added", "face_id": face_id}

@app.get("/tenants/{tenant_id}/faces")
def list_faces_endpoint(tenant_id: str):
    faces = list_face_records(tenant_id)
    return {"faces": faces}

@app.put("/tenants/{tenant_id}/faces/{face_id}")
async def update_face_endpoint(
    tenant_id: str,
    face_id: int,
    camera_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    metadata: dict = Body({})
):
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
    update_face_record_helper(face_id, tenant_id, camera_id, name, embedding, metadata)
    return {"message": "Face updated successfully."}

@app.delete("/tenants/{tenant_id}/faces/{face_id}")
def remove_face(tenant_id: str, face_id: int):
    delete_face_record(face_id, tenant_id)
    return {"message": "Face deleted."}

######################################################################
# 6) Video Deletion and Update Endpoints
######################################################################
@app.delete("/videos/{video_id}")
def delete_video(video_id: int):
    success = delete_video_record(video_id)
    if not success:
        raise HTTPException(status_code=404, detail="Video not found.")
    videos_info.pop(video_id, None)
    return {"message": f"Video {video_id} deleted successfully."}

@app.put("/videos/{video_id}")
async def update_video(
    video_id: int,
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

    if file:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        cap = cv2.VideoCapture(file_location)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Cannot open video file: {file.filename}")
        size_bytes = os.path.getsize(file_location)
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fps = original_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        update_fields["filename"] = file_location
        update_fields["size"] = size_bytes
        update_fields["fps"] = fps
        update_fields["total_frames"] = total_frames
        update_fields["duration"] = duration

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    update_video_record(video_id, update_fields)
    if video_id in videos_info:
        videos_info[video_id].update(update_fields)
    return {"message": f"Video {video_id} updated successfully."}

######################################################################
# 7) External Trigger Testing Endpoint
######################################################################
@app.post("/trigger-event")
async def trigger_event(payload: dict = Body(...)):
    tenant_id = payload.get("tenant_id")
    camera_id = payload.get("camera_id")
    event_url = payload.get("event_url")
    if not tenant_id or not camera_id or not event_url:
        raise HTTPException(status_code=400, detail="tenant_id, camera_id, and event_url are required.")
    status_code, response_text = await trigger_external_event(event_url, payload)
    return {"status_code": status_code, "response_text": response_text}
