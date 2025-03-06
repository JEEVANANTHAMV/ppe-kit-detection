import os
import cv2
import time
import asyncio
import sqlite3
import datetime
import numpy as np
# import face_recognition
import redis.asyncio as aioredis
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

app = FastAPI()
confidence_threshold = int(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH)
# model.to("cuda")  # run YOLO on GPU

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

VIOLATION_DIR = "violation_images"
os.makedirs(VIOLATION_DIR, exist_ok=True)
redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)

known_faces = {}
next_person_id = 1

VIOLATION_THRESHOLDS = {
    "NO-Hardhat": 10,
    "NO-Mask": 20,
    "NO-Safety Vest": 44
}

def init_db():
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS violations (
            video_name TEXT,
            violation_timestamp REAL,
            person_number INTEGER,
            violation_type TEXT,
            violation_image_path TEXT
        )"""
    )
    conn.commit()
    conn.close()

init_db()

def save_violation_to_db(video_name, violation_timestamp, person_number, violation_type, violation_image_path):
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO violations VALUES (?, ?, ?, ?, ?)",
        (video_name, violation_timestamp, person_number, violation_type, violation_image_path),
    )
    conn.commit()
    conn.close()

"""
We'll store metadata and processing status in memory for easy reference:
    videos_info = {
       1: {
           "video_name": "video1.mp4",
           "filename": "uploads/video1.mp4",
           "size": 12345678,   # in bytes
           "fps": 30.0,
           "total_frames": 900,  # approximate
           "duration": 30.0,
           "status": "uploaded"  # or "processing" or "done"
           "frames_processed": 0,
           "violations_detected": 0,
           "persons_involved": 0,
       },
       2: {...},
       ...
    }
"""
videos_info = {}
video_id_counter = 1
REQUIRED_ITEMS = {"No-Mask", "No-Safety-Vest", "No-Hardhat"}

def detect_objects(image):
    """
    YOLOv8 detection: returns boxes, results, annotated_image (RGB),
    colors, and class_confidences dict
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, verbose=False)[0]
    annotated_image = image_rgb.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    boxes = results.boxes
    class_confidences = {}
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results.names[cls_id]
        if class_name not in class_confidences:
            class_confidences[class_name] = []
        class_confidences[class_name].append(conf)
        violators = []
        if class_name == "Person":
                person_id = f"Person_{i}"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detected_items = set()
                for j, sub_box in enumerate(boxes):
                    sub_cls_id = int(sub_box.cls[0])
                    sub_class_name = results.names[sub_cls_id]
                    sx1, sy1, sx2, sy2 = map(int, sub_box.xyxy[0])

                    if x1 <= sx1 <= x2 and y1 <= sy1 <= y2:
                        detected_items.add(sub_class_name)

                missing_items = REQUIRED_ITEMS - detected_items
                if missing_items:
                    # face_image = extract_face(image, box)
                    # log_violation(person_id, face_image)
                    violators.append((person_id, missing_items))

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated_image, "Violation", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print("Ravan ", violators)
    return boxes, results, annotated_image, colors, class_confidences

def check_violations(class_confidences, threshold=0.5):
    """
    If any "NO-*" class has confidence > threshold, consider it a violation.
    Returns list of classes that are in violation.
    """
    violations_found = []
    for class_name, conf_list in class_confidences.items():
        if class_name.startswith("NO-"):
            if any(conf > threshold for conf in conf_list):
                violations_found.append(class_name)
    return violations_found

def save_annotated_plot(
    original_bgr_image,
    boxes,
    class_names,
    annotated_image_rgb,
    colors,
    confidence_threshold,
    video_source
):
    """
    Saves the annotated results to a PNG file. Returns path.
    """
    original_image_rgb = cv2.cvtColor(original_bgr_image, cv2.COLOR_BGR2RGB)
    class_labels = {}
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        if confidence > confidence_threshold:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            color = colors[class_id % len(colors)].tolist()
            cv2.rectangle(annotated_image_rgb, (x1, y1), (x2, y2), color, 2)
            class_labels[class_name] = color

    plt.figure(figsize=(12, 6))
    # Original
    plt.subplot(1, 2, 1)
    plt.title('Original Frame')
    plt.imshow(original_image_rgb)
    plt.axis('off')
    # Annotated
    plt.subplot(1, 2, 2)
    plt.title('Detected Objects')
    plt.imshow(annotated_image_rgb)
    plt.axis('off')

    legend_handles = []
    for class_name, color in class_labels.items():
        normalized_color = np.array(color) / 255.0
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                       markerfacecolor=normalized_color, markersize=10)
        )
    if legend_handles:
        plt.legend(handles=legend_handles, loc='upper right', title='Classes')

    plt.tight_layout()

    output_dir = os.getenv("OUTPUT_DIR", "./violation_images")
    os.makedirs(output_dir, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_filename = f"detected_objects_{timestamp_str}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Annotated result saved to: {output_path}")
    return output_path

async def process_video_stream(video_id: int):
    """
    Reads frames from the video, does YOLO detection at ~20 FPS,
    logs violations, updates videos_info in real-time.
    """
    if video_id not in videos_info:
        return

    video_meta = videos_info[video_id]
    video_path = video_meta["filename"]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_path}")
        video_meta["status"] = "error"
        return

    video_meta["status"] = "processing"
    
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip_interval = max(1, int(original_fps // 20))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue

        boxes, results, annotated_img, colors, class_confidences = await asyncio.to_thread(detect_objects, frame)
        violations_found = check_violations(class_confidences, threshold=0.5)
        if violations_found:
            annotated_path = save_annotated_plot(
                original_bgr_image=frame,
                boxes=boxes,
                class_names=results.names,
                annotated_image_rgb=annotated_img,
                colors=colors,
                confidence_threshold=0.5,
                video_source=video_path
            )

            for vio in violations_found:
                save_violation_to_db(
                    video_name=video_meta["video_name"],
                    violation_timestamp=time.time(),
                    person_number=None,
                    violation_type=vio,
                    violation_image_path=annotated_path
                )
            video_meta["violations_detected"] += len(violations_found)
        video_meta["frames_processed"] += frame_skip_interval

        await asyncio.sleep(0)

    cap.release()
    video_meta["status"] = "done"
    print(f"[INFO] Finished processing {video_meta['video_name']}.")

@app.post("/videos")
async def upload_videos(files: List[UploadFile] = File(...)):
    """
    1) POST /videos to upload multiple video files (up to 5).
    If user sends more than 5, we only accept the first 5.
    """
    global video_id_counter
    accepted_files = files[:5]  # only first 5

    if not accepted_files:
        return {"message": "No files received."}

    responses = []
    for f in accepted_files:
        file_location = os.path.join(UPLOAD_DIR, f.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await f.read())

        cap = cv2.VideoCapture(file_location)
        if not cap.isOpened():
            responses.append({"filename": f.filename, "error": "Cannot read video."})
            continue
        size_bytes = os.path.getsize(file_location)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        v_id = video_id_counter
        videos_info[v_id] = {
            "video_name": f.filename,
            "filename": file_location,
            "size": size_bytes,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "status": "uploaded",
            "frames_processed": 0,
            "violations_detected": 0,
            "persons_involved": 0
        }
        video_id_counter += 1

        responses.append({"video_id": v_id, "filename": f.filename, "size": size_bytes,
                          "fps": fps, "duration": duration})

    return {"accepted_videos": responses}


@app.get("/videos")
def list_videos():
    """
    2) GET /videos to list all videos with info: size, fps, duration, status, etc.
    """
    all_vids = []
    for vid_id, meta in videos_info.items():
        all_vids.append({
            "video_id": vid_id,
            "video_name": meta["video_name"],
            "size": meta["size"],
            "fps": meta["fps"],
            "total_frames": meta["total_frames"],
            "duration": meta["duration"],
            "status": meta["status"]
        })
    return {"videos": all_vids}


@app.post("/process")
async def process_videos(video_ids: Union[List[int], str] = Body(...)):
    """
    3) POST /process
       - if body is "[*]", process all videos in parallel (limit to those with status 'uploaded' or 'done' but re-run if you want).
       - if body is a list of IDs [1,2], process those specifically.
    """
    if isinstance(video_ids, str) and video_ids.strip() == "*":
        vids_to_run = list(videos_info.keys())
    elif isinstance(video_ids, list):
        vids_to_run = video_ids
    else:
        raise HTTPException(status_code=400, detail="Invalid input. Must be '*' or list of IDs.")

    tasks = []
    for vid in vids_to_run:
        if vid not in videos_info:
            continue
        tasks.append(asyncio.create_task(process_video_stream(vid)))

    if not tasks:
        return {"message": "No valid videos to process."}

    return {"message": f"Triggered processing for videos: {vids_to_run}"}


@app.get("/status")
def get_status():
    """
    4) GET /status
    Return real-time status: frames processed, # violations, # persons involved
    """
    status_list = []
    for vid_id, meta in videos_info.items():
        status_list.append({
            "video_id": vid_id,
            "video_name": meta["video_name"],
            "status": meta["status"],
            "frames_processed": meta["frames_processed"],
            "violations_detected": meta["violations_detected"],
            "persons_involved": meta["persons_involved"]
        })
    return {"videos_status": status_list}


@app.get("/stats")
def get_stats():
    """
    5) GET /stats
    - total video duration
    - total processed duration
    - number of persons involved in violations
    - highest violation count for each video
    """
    total_duration = sum(v["duration"] for v in videos_info.values())
    total_processed_frames = sum(v["frames_processed"] for v in videos_info.values())
   
    total_processed_duration = 0
    for meta in videos_info.values():
        fps = meta["fps"] if meta["fps"] > 0 else 30
        processed_dur = (meta["frames_processed"] / fps)
        total_processed_duration += processed_dur

    total_persons_involved = sum(v["persons_involved"] for v in videos_info.values())

    violation_counts = {}
    for vid_id, meta in videos_info.items():
        violation_counts[vid_id] = meta["violations_detected"]

    return {
        "total_video_duration": total_duration,
        "total_processed_duration": total_processed_duration,
        "total_persons_involved_in_violation": total_persons_involved,
        "violation_counts_per_video": violation_counts
    }


# --------------------------------------------------
# HOW TO RUN:
# uvicorn this_script:app --reload
# Then test endpoints:
# 1) POST /videos (multipart form-data: files=)
# 2) GET /videos
# 3) POST /process with body = "*" or [1,2]
# 4) GET /status
# 5) GET /stats
# --------------------------------------------------
