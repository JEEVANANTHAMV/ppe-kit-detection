import os
import json
import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import httpx
from deepface import DeepFace
from ultralytics import YOLO
import logging
import uuid
import faiss

# Try to import psycopg2 for Postgres support
try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Database fallback path for SQLite
DB_PATH = os.getenv("SQLITE_DB_PATH", "violations.db")

def generate_uuid():
    return str(uuid.uuid4())

# ----------------- Database Connection Helpers ----------------- #
def get_connection():
    """
    Returns a tuple (conn, db_type). Uses Postgres if DATABASE_URL is set and psycopg2 is available;
    otherwise connects to local SQLite.
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url and psycopg2 is not None:
        try:
            conn = psycopg2.connect(database_url)
            return conn, "postgres"
        except Exception as e:
            logging.error("Error connecting to Postgres: %s", e)
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    return conn, "sqlite"

def format_query(query, db_type):
    """
    Converts '?' placeholders to '%s' for Postgres.
    """
    if db_type == "postgres":
        return query.replace("?", "%s")
    return query

# ----------------- YOLO Initialization ----------------- #
MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH)
gpu_semaphore = None

try:
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
        import asyncio
        gpu_semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent GPU operations
except Exception:
    pass

# ----------------- Required Items ----------------- #
REQUIRED_ITEMS = {"NO-Mask", "NO-Safety-Vest", "NO-Hardhat"}

# ----------------- Database Initialization ----------------- #
def init_db():
    conn, db_type = get_connection()
    c = conn.cursor()
    # Tenant configuration table
    query = """
    CREATE TABLE IF NOT EXISTS tenant_config (
        tenant_id TEXT PRIMARY KEY,
        similarity_threshold REAL,
        no_mask_threshold INTEGER,
        no_safety_vest_threshold INTEGER,
        no_hardhat_threshold INTEGER,
        external_trigger_url TEXT
    )
    """
    c.execute(format_query(query, db_type))
    # Videos table – note: using TEXT for video_id (UUID) and adding processing status columns.
    query = """
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        is_live INTEGER NOT NULL,
        filename TEXT,
        stream_url TEXT,
        size INTEGER DEFAULT 0,
        fps REAL DEFAULT 0,
        total_frames INTEGER DEFAULT 0,
        duration REAL DEFAULT 0,
        status TEXT DEFAULT 'uploaded',
        frames_processed INTEGER DEFAULT 0,
        violations_detected INTEGER DEFAULT 0,
        UNIQUE(tenant_id, camera_id)
    )
    """
    c.execute(format_query(query, db_type))
    # Faces table
    query = """
    CREATE TABLE IF NOT EXISTS faces (
        face_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        name TEXT,
        embedding TEXT,
        metadata TEXT
    )
    """
    c.execute(format_query(query, db_type))
    # Violations table
    query = """
    CREATE TABLE IF NOT EXISTS violations (
        id TEXT PRIMARY KEY,
        tenant_id TEXT,
        camera_id TEXT,
        violation_timestamp REAL,
        face_id TEXT,
        violation_type TEXT,
        violation_image_path TEXT,
        details TEXT
    )
    """
    c.execute(format_query(query, db_type))
    conn.commit()
    conn.close()

# ----------------- Videos Table Operations ----------------- #
def check_camera_exists(tenant_id: str, camera_id: str) -> bool:
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT COUNT(*) FROM videos WHERE tenant_id = ? AND camera_id = ?"
    c.execute(format_query(query, db_type), (tenant_id, camera_id))
    count = c.fetchone()[0]
    conn.close()
    return count > 0

def insert_video_record(
    tenant_id: str,
    camera_id: str,
    is_live: bool,
    filename: str = None,
    stream_url: str = None,
    size: int = 0,
    fps: float = 0,
    total_frames: int = 0,
    duration: float = 0
) -> str:
    conn, db_type = get_connection()
    c = conn.cursor()
    video_id = generate_uuid()
    status = "registered" if is_live else "uploaded"
    query = """
    INSERT INTO videos (video_id, tenant_id, camera_id, is_live, filename, stream_url, size, fps, total_frames, duration, status, frames_processed, violations_detected)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
    """
    c.execute(format_query(query, db_type), (
        video_id,
        tenant_id,
        camera_id,
        1 if is_live else 0,
        filename,
        stream_url,
        size,
        fps,
        total_frames,
        duration,
        status
    ))
    conn.commit()
    conn.close()
    return video_id

def list_video_records():
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT video_id, tenant_id, camera_id, is_live, filename, stream_url, status, size, fps, total_frames, duration, frames_processed, violations_detected FROM videos"
    c.execute(format_query(query, db_type))
    rows = c.fetchall()
    conn.close()
    videos = []
    for row in rows:
        videos.append({
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
        })
    return videos

def get_video_record(video_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT video_id, tenant_id, camera_id, is_live, filename, stream_url, status, size, fps, total_frames, duration, frames_processed, violations_detected FROM videos WHERE video_id = ?"
    c.execute(format_query(query, db_type), (video_id,))
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

def update_video_record(video_id: str, fields: dict) -> bool:
    if not fields:
        return False
    conn, db_type = get_connection()
    c = conn.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in fields.keys()])
    values = list(fields.values())
    values.append(video_id)
    query = f"UPDATE videos SET {set_clause} WHERE video_id = ?"
    c.execute(format_query(query, db_type), tuple(values))
    conn.commit()
    conn.close()
    return True

def delete_video_record(video_id: str) -> bool:
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT filename FROM videos WHERE video_id = ?"
    c.execute(format_query(query, db_type), (video_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    filename = row[0]
    query = "DELETE FROM videos WHERE video_id = ?"
    c.execute(format_query(query, db_type), (video_id,))
    conn.commit()
    conn.close()
    if filename and os.path.exists(filename):
        os.remove(filename)
    return True

def update_video_status(video_id: str, status: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "UPDATE videos SET status = ? WHERE video_id = ?"
    c.execute(format_query(query, db_type), (status, video_id))
    conn.commit()
    conn.close()

def increment_frames_processed(video_id: str, frames: int):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "UPDATE videos SET frames_processed = frames_processed + ? WHERE video_id = ?"
    c.execute(format_query(query, db_type), (frames, video_id))
    conn.commit()
    conn.close()

def increment_violations_detected(video_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "UPDATE videos SET violations_detected = violations_detected + 1 WHERE video_id = ?"
    c.execute(format_query(query, db_type), (video_id,))
    conn.commit()
    conn.close()

# ----------------- Tenant Config Operations ----------------- #
def get_tenant_config(tenant_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url FROM tenant_config WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (tenant_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "similarity_threshold": row[0],
            "no_mask_threshold": row[1],
            "no_safety_vest_threshold": row[2],
            "no_hardhat_threshold": row[3],
            "external_trigger_url": row[4]
        }
    return None

def add_or_update_tenant_config(tenant_id, similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url):
    existing = get_tenant_config(tenant_id)
    conn, db_type = get_connection()
    c = conn.cursor()
    if existing:
        query = "UPDATE tenant_config SET similarity_threshold = ?, no_mask_threshold = ?, no_safety_vest_threshold = ?, no_hardhat_threshold = ?, external_trigger_url = ? WHERE tenant_id = ?"
        c.execute(format_query(query, db_type), (similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url, tenant_id))
    else:
        query = "INSERT INTO tenant_config (tenant_id, similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url) VALUES (?, ?, ?, ?, ?, ?)"
        c.execute(format_query(query, db_type), (tenant_id, similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url))
    conn.commit()
    conn.close()

def delete_tenant_config(tenant_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "DELETE FROM tenant_config WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (tenant_id,))
    conn.commit()
    conn.close()

# ----------------- Faces Table Operations ----------------- #
def add_face_record(tenant_id, camera_id, name, embedding, metadata, face_id) -> str:
    conn, db_type = get_connection()
    c = conn.cursor()
    
    # Check if face_id already exists
    query = "SELECT COUNT(*) FROM faces WHERE face_id = ?"
    c.execute(format_query(query, db_type), (face_id,))
    if c.fetchone()[0] > 0:
        conn.close()
        raise ValueError(f"Face ID {face_id} already exists")
        
    embedding_json = json.dumps(embedding)
    metadata_json = json.dumps(metadata)
    query = "INSERT INTO faces (face_id, tenant_id, camera_id, name, embedding, metadata) VALUES (?, ?, ?, ?, ?, ?)"
    c.execute(format_query(query, db_type), (face_id, tenant_id, camera_id, name, embedding_json, metadata_json))
    conn.commit()
    conn.close()
    return face_id

def update_face_record(face_id, tenant_id, camera_id, name, embedding, metadata):
    conn, db_type = get_connection()
    c = conn.cursor()
    embedding_json = json.dumps(embedding)
    metadata_json = json.dumps(metadata)
    query = "UPDATE faces SET camera_id = ?, name = ?, embedding = ?, metadata = ? WHERE face_id = ? AND tenant_id = ?"
    c.execute(format_query(query, db_type), (camera_id, name, embedding_json, metadata_json, face_id, tenant_id))
    conn.commit()
    conn.close()

def delete_face_record(face_id, tenant_id):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "DELETE FROM faces WHERE face_id = ? AND tenant_id = ?"
    c.execute(format_query(query, db_type), (face_id, tenant_id))
    conn.commit()
    conn.close()

def list_face_records(tenant_id):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT face_id, camera_id, name, embedding, metadata FROM faces WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (tenant_id,))
    rows = c.fetchall()
    conn.close()
    faces = []
    for row in rows:
        faces.append({
            "face_id": row[0],
            "camera_id": row[1],
            "name": row[2],
            "embedding": json.loads(row[3]),
            "metadata": json.loads(row[4])
        })
    return faces

# ----------------- Violations Table Operations ----------------- #
def save_violation_to_db(
    tenant_id,
    camera_id,
    violation_timestamp,
    face_id,
    violation_type,
    violation_image_path,
    details=""
):
    id = generate_uuid()
    conn, db_type = get_connection()
    c = conn.cursor()
    query = """
    INSERT INTO violations (id, tenant_id, camera_id, violation_timestamp, face_id, violation_type, violation_image_path, details)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    c.execute(format_query(query, db_type), (
        id,
        tenant_id,
        camera_id,
        violation_timestamp,
        face_id,
        violation_type,
        violation_image_path,
        details
    ))
    conn.commit()
    conn.close()

# ----------------- Detection & Recognition ----------------- #
async def detect_objects_with_semaphore(image):
    if gpu_semaphore:
        async with gpu_semaphore:
            return detect_objects(image)
    return detect_objects(image)

def detect_objects(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, verbose=False)[0]
    annotated_image = image_rgb.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    boxes = results.boxes
    class_confidences = {}
    violation_faces = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results.names[cls_id]
        class_confidences.setdefault(class_name, []).append(conf)
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
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_image, "Violation", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                violation_faces.append((person_id, (x1, y1, x2, y2)))
    return boxes, results, annotated_image, colors, class_confidences, violation_faces

def check_violations(class_confidences, threshold=0.5):
    violations_found = []
    for class_name, conf_list in class_confidences.items():
        if class_name.startswith("NO-") and any(conf > threshold for conf in conf_list):
            violations_found.append(class_name)
    return violations_found

def save_annotated_plot(original_bgr_image, boxes, class_names, annotated_image_rgb, colors, confidence_threshold, video_source):
    output_dir = os.getenv("OUTPUT_DIR", "./violation_images")
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_filename = f"detected_objects_{timestamp_str}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Create a side-by-side image
    h, w = original_bgr_image.shape[:2]
    combined_image = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined_image[:, :w] = original_bgr_image
    combined_image[:, w:] = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
    
    # Add class labels
    y_offset = 30
    for class_name, color in class_names.items():
        cv2.putText(combined_image, class_name, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
    
    cv2.imwrite(output_path, combined_image)
    logging.info(f"[INFO] Annotated result saved to: {output_path}")
    return output_path

def extract_face_embedding(image, face_box):
    x1, y1, x2, y2 = face_box
    
    # Validate and clamp coordinates to image dimensions
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    # Check if box is valid
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        logging.error("Invalid face box dimensions")
        return None
        
    face_crop = image[y1:y2, x1:x2]
    try:
        embeddings = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)
        if isinstance(embeddings, list) and len(embeddings) > 0:
            return embeddings[0]["embedding"]
    except Exception as e:
        logging.error("extract_face_embedding error: %s", e)
    return None

def calculate_cosine_distance(emb1, emb2):
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Return 0 similarity for zero vectors
    similarity = dot / (norm1 * norm2)
    return similarity  # Return similarity directly

def compare_embeddings(embedding1, embedding2, threshold):
    similarity = calculate_cosine_distance(embedding1, embedding2)
    return similarity >= threshold  # Compare similarity directly against threshold

# ----------------- External Trigger ----------------- #
async def trigger_external_event(event_url, payload):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(event_url, json=payload, timeout=10)
            return response.status_code, response.text
        except Exception as e:
            logging.error("Error triggering external event: %s", e)
            return None, str(e)

# Global FAISS index per tenant
face_indices = {}
face_id_maps = {}

def build_face_index(tenant_id: str, faces: list):
    if not faces:
        return
    
    embeddings = []
    face_ids = []
    for face in faces:
        embedding = np.array(face["embedding"], dtype=np.float32)
        embeddings.append(embedding)
        face_ids.append(face["face_id"])
    
    embeddings = np.array(embeddings)
    d = embeddings.shape[1]  # embedding dimension
    
    # Build FAISS index
    index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize vectors for cosine similarity
    index.add(embeddings)
    
    face_indices[tenant_id] = index
    face_id_maps[tenant_id] = face_ids

def find_matching_faces(tenant_id: str, embedding: np.ndarray, similarity_threshold: float) -> list:
    if tenant_id not in face_indices:
        all_faces = list_face_records(tenant_id)
        build_face_index(tenant_id, all_faces)
    
    if tenant_id not in face_indices:
        return []
        
    query = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(query)
    
    D, I = face_indices[tenant_id].search(query, k=10)  # Get top 10 matches
    matches = []
    
    for i, dist in zip(I[0], D[0]):
        if dist >= similarity_threshold and i < len(face_id_maps[tenant_id]):
            matches.append(face_id_maps[tenant_id][i])
            
    return matches
