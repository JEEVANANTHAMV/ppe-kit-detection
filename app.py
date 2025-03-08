import os
import cv2
import time
import json
import asyncio
import datetime
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

from helper import (
    init_db,
    check_camera_exists,
    insert_video_record,
    get_tenant_config,
    list_face_records,
    compare_embeddings,
    detect_objects,
    check_violations,
    save_annotated_plot,
    save_violation_to_db,
    extract_face_embedding,
    trigger_external_event,
    # Face management
    add_face_record,
    update_face_record,
    delete_face_record,
    list_face_records,
    # Tenant config
    add_or_update_tenant_config,
    delete_tenant_config
)

app = FastAPI(title="Safety Violation Detector")
init_db()

# Global in-memory structures
videos_info = {}  # video_id -> metadata (for runtime tracking)
violation_counters = {}  # (tenant_id, camera_id, face_id) -> int
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
      - If is_live=True, must provide stream_url (no file).
      - If is_live=False, must provide a file (no stream_url).
      - (tenant_id, camera_id) must be unique in the database.
    """
    # 1) Check if (tenant_id, camera_id) already exists
    if check_camera_exists(tenant_id, camera_id):
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' already exists for tenant '{tenant_id}'."
        )

    # 2) If live stream
    if is_live:
        if not stream_url or file is not None:
            raise HTTPException(
                status_code=400,
                detail="For a live feed (is_live=True), provide stream_url only (no file)."
            )
        video_id = insert_video_record(
            tenant_id=tenant_id,
            camera_id=camera_id,
            is_live=True,
            stream_url=stream_url
        )
        # We'll store minimal metadata for a live stream
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

    # 3) Otherwise, offline video
    else:
        if file is None or stream_url is not None:
            raise HTTPException(
                status_code=400,
                detail="For an offline video (is_live=False), provide a file only (no stream_url)."
            )
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Extract metadata with OpenCV
        cap = cv2.VideoCapture(file_location)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot open or read video file: {file.filename}"
            )
        size_bytes = os.path.getsize(file_location)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
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
# 2) Listing & Status Endpoints
######################################################################

@app.get("/videos")
def list_videos():
    """
    Lists all videos/streams currently known in-memory.
    (In production, you might want to query the DB instead.)
    """
    result = []
    for vid_id, meta in videos_info.items():
        result.append({
            "video_id": vid_id,
            "tenant_id": meta["tenant_id"],
            "camera_id": meta["camera_id"],
            "is_live": meta["is_live"],
            "filename": meta["filename"],
            "stream_url": meta["stream_url"],
            "status": meta["status"],
            "size": meta["size"],
            "fps": meta["fps"],
            "total_frames": meta["total_frames"],
            "duration": meta["duration"]
        })
    return {"videos": result}


@app.get("/status")
def get_status():
    """
    Returns real-time status for each video/stream.
    """
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
    """
    Returns aggregated stats across all videos/streams:
      - total_video_duration
      - total_processed_duration
      - violation_counts_per_video
    """
    total_duration = sum(v["duration"] for v in videos_info.values())
    total_processed_duration = 0
    violation_counts = {}
    for vid_id, meta in videos_info.items():
        fps = meta["fps"] if meta["fps"] > 0 else 30
        processed_dur = meta["frames_processed"] / fps
        total_processed_duration += processed_dur
        violation_counts[vid_id] = meta["violations_detected"]
    return {
        "total_video_duration": total_duration,
        "total_processed_duration": total_processed_duration,
        "violation_counts_per_video": violation_counts
    }


######################################################################
# 3) Processing Videos/Streams
######################################################################

async def process_video_stream(video_id: int):
    """
    Background task to process a single video or live stream:
      1) Grab frames from the source (file or stream).
      2) YOLO detection to find persons missing safety items.
      3) Attempt to extract face embeddings (DeepFace).
      4) If face recognized, increment violation counters.
      5) If counter >= threshold, log to DB + optional external event.
    """
    if video_id not in videos_info:
        return

    meta = videos_info[video_id]
    tenant_id = meta["tenant_id"]
    camera_id = meta["camera_id"]
    config = get_tenant_config(tenant_id)
    if not config:
        print(f"[ERROR] No tenant configuration found for tenant {tenant_id}. Aborting.")
        meta["status"] = "error"
        return
    similarity_threshold = config["similarity_threshold"]
    violation_threshold = config["violation_threshold"]
    external_trigger_url = config["external_trigger_url"]

    # Open source
    cap = cv2.VideoCapture(meta["stream_url"] if meta["is_live"] else meta["filename"])
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source for video_id {video_id}")
        meta["status"] = "error"
        return

    meta["status"] = "processing"
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip_interval = max(1, int(original_fps // 20))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Skip frames to aim ~20 FPS
        if frame_count % frame_skip_interval != 0:
            continue

        # 1) Detect objects + persons with missing items
        boxes, results, annotated_img, colors, class_confidences, violation_faces = await asyncio.to_thread(
            detect_objects, frame
        )
        # 2) Check for "NO-" classes
        violations_found = check_violations(class_confidences, threshold=0.5)
        if violations_found:
            # 2a) Save annotated frame
            annotated_path = save_annotated_plot(
                original_bgr_image=frame,
                boxes=boxes,
                class_names=results.names,
                annotated_image_rgb=annotated_img,
                colors=colors,
                confidence_threshold=0.5,
                video_source=meta["stream_url"] if meta["is_live"] else meta["filename"]
            )
            # 2b) For each violating person, try face recognition
            for person_id, (x1, y1, x2, y2) in violation_faces:
                # Attempt face extraction
                embedding = extract_face_embedding(frame, (x1, y1, x2, y2))
                if embedding is None:
                    # Face not visible or can't be extracted
                    continue

                # Try matching with known faces for this tenant
                matched_face_id = None
                all_faces = list_face_records(tenant_id)
                for face_rec in all_faces:
                    # We only consider faces that are relevant to the same camera
                    if face_rec["camera_id"] != camera_id:
                        continue
                    known_embedding = face_rec["embedding"]
                    if compare_embeddings(embedding, known_embedding, similarity_threshold):
                        matched_face_id = face_rec["face_id"]
                        break

                if not matched_face_id:
                    # Face recognized for no known user => skip
                    continue

                # 2c) Increment violation counter
                key = (tenant_id, camera_id, matched_face_id)
                violation_counters[key] = violation_counters.get(key, 0) + 1

                # 2d) If threshold is reached, escalate
                if violation_counters[key] >= violation_threshold:
                    # Save to DB
                    save_violation_to_db(
                        tenant_id=tenant_id,
                        camera_id=camera_id,
                        violation_timestamp=time.time(),
                        face_id=matched_face_id,
                        violation_type="Repeated Violation",
                        violation_image_path=annotated_path,
                        details=json.dumps({"person_id": person_id, "count": violation_counters[key]})
                    )
                    # Reset counter
                    violation_counters[key] = 0
                    # Increment local count
                    meta["violations_detected"] += 1
                    # Optional external event
                    if external_trigger_url:
                        payload = {
                            "tenant_id": tenant_id,
                            "camera_id": camera_id,
                            "face_id": matched_face_id,
                            "violation_count": violation_threshold,
                            "timestamp": time.time()
                        }
                        asyncio.create_task(trigger_external_event(external_trigger_url, payload))

        meta["frames_processed"] += frame_skip_interval
        await asyncio.sleep(0)

    cap.release()
    meta["status"] = "done"
    print(f"[INFO] Finished processing video_id={video_id} (tenant={tenant_id}, camera={camera_id}).")


@app.post("/process")
async def process_videos(
    video_ids: Union[List[int], str] = Body(...),
    background_tasks: BackgroundTasks = None
):
    """
    Triggers processing of the specified videos/streams.
    Provide a list of video IDs or "*" to process all.
    """
    if isinstance(video_ids, str) and video_ids.strip() == "*":
        vids_to_run = list(videos_info.keys())
    elif isinstance(video_ids, list):
        vids_to_run = video_ids
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Must be '*' or list of video IDs."
        )

    if not vids_to_run:
        return {"message": "No videos to process."}

    triggered = []
    for vid_id in vids_to_run:
        if vid_id in videos_info:
            # Queue the background task
            background_tasks.add_task(process_video_stream, vid_id)
            triggered.append(vid_id)

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
def create_config(tenant_id: str, body: dict = Body(...)):
    similarity_threshold = body.get("similarity_threshold")
    violation_threshold = body.get("violation_threshold")
    external_trigger_url = body.get("external_trigger_url", "")
    if similarity_threshold is None or violation_threshold is None:
        raise HTTPException(status_code=400, detail="Missing threshold values.")
    add_or_update_tenant_config(tenant_id, similarity_threshold, violation_threshold, external_trigger_url)
    return {"message": "Configuration created/updated."}


@app.put("/tenants/{tenant_id}/config")
def update_config(tenant_id: str, body: dict = Body(...)):
    similarity_threshold = body.get("similarity_threshold")
    violation_threshold = body.get("violation_threshold")
    external_trigger_url = body.get("external_trigger_url", "")
    if similarity_threshold is None or violation_threshold is None:
        raise HTTPException(status_code=400, detail="Missing threshold values.")
    add_or_update_tenant_config(tenant_id, similarity_threshold, violation_threshold, external_trigger_url)
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
    camera_id: str = Body(...),
    name: str = Body(...),
    file: UploadFile = File(...),
    metadata: dict = Body({})
):
    """
    Adds a new face record for a tenant & camera.
    The uploaded image is used to extract a face embedding.
    """
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

    face_id = add_face_record(tenant_id, camera_id, name, embedding, metadata)
    return {"message": "Face added", "face_id": face_id}


@app.get("/tenants/{tenant_id}/faces")
def list_faces_endpoint(tenant_id: str):
    faces = list_face_records(tenant_id)
    return {"faces": faces}


@app.put("/tenants/{tenant_id}/faces/{face_id}")
async def update_face_endpoint(
    tenant_id: str,
    face_id: int,
    camera_id: str = Body(...),
    name: str = Body(...),
    file: UploadFile = File(...),
    metadata: dict = Body({})
):
    """
    Updates an existing face record.
    """
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

    update_face_record(face_id, tenant_id, camera_id, name, embedding, metadata)
    return {"message": "Face updated successfully."}


@app.delete("/tenants/{tenant_id}/faces/{face_id}")
def remove_face(tenant_id: str, face_id: int):
    delete_face_record(face_id, tenant_id)
    return {"message": "Face deleted."}


######################################################################
# 6) External Trigger Testing
######################################################################

@app.post("/trigger-event")
async def trigger_event(payload: dict = Body(...)):
    """
    Manually trigger an external event by providing a payload with:
      - tenant_id
      - camera_id
      - event_url
    """
    tenant_id = payload.get("tenant_id")
    camera_id = payload.get("camera_id")
    event_url = payload.get("event_url")
    if not tenant_id or not camera_id or not event_url:
        raise HTTPException(
            status_code=400,
            detail="tenant_id, camera_id, and event_url are required."
        )
    status_code, response_text = await trigger_external_event(event_url, payload)
    return {"status_code": status_code, "response_text": response_text}
