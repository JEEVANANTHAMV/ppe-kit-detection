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
try:
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
except Exception:
    pass

# ----------------- Required Items ----------------- #
REQUIRED_ITEMS = {"No-Mask", "No-Safety-Vest", "No-Hardhat"}

# ----------------- Database Initialization ----------------- #
def init_db():
    conn, db_type = get_connection()
    c = conn.cursor()
    # Create tenant_config table with separate columns for each violation threshold
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
    # Videos table
    query = """
    CREATE TABLE IF NOT EXISTS videos (
        video_id SERIAL PRIMARY KEY,
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
        UNIQUE(tenant_id, camera_id)
    )
    """
    c.execute(format_query(query, db_type))
    # Faces table
    query = """
    CREATE TABLE IF NOT EXISTS faces (
        face_id SERIAL PRIMARY KEY,
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
        id SERIAL PRIMARY KEY,
        tenant_id TEXT,
        camera_id TEXT,
        violation_timestamp REAL,
        face_id INTEGER,
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
) -> int:
    conn, db_type = get_connection()
    c = conn.cursor()
    video_id = generate_uuid()
    query = """
    INSERT INTO videos (video_id, tenant_id, camera_id, is_live, filename, stream_url, size, fps, total_frames, duration, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    status = "registered" if is_live else "uploaded"
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
    query = "SELECT video_id, tenant_id, camera_id, is_live, filename, stream_url, status, size, fps, total_frames, duration FROM videos"
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
            "duration": row[10]
        })
    return videos

def get_video_record(video_id: int):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT video_id, tenant_id, camera_id, is_live, filename, stream_url, status, size, fps, total_frames, duration FROM videos WHERE video_id = ?"
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
            "duration": row[10]
        }
    return None

def update_video_record(video_id: int, fields: dict) -> bool:
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

def delete_video_record(video_id: int) -> bool:
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

# ----------------- Tenant Config Operations ----------------- #
def get_tenant_config(tenant_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url FROM tenant_config WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (tenant_id,))
    row = c.fetchone()
    conn.close()
    if row:
        print(row)
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
def add_face_record(tenant_id, camera_id, name, embedding, metadata) -> int:
    conn, db_type = get_connection()
    c = conn.cursor()
    embedding_json = json.dumps(embedding)
    metadata_json = json.dumps(metadata)
    face_id = generate_uuid()
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
    print("Execute")
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
    print("done")
    conn.commit()
    conn.close()


# ----------------- Detection & Recognition ----------------- #
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
    plt.subplot(1, 2, 1)
    plt.title("Original Frame")
    plt.imshow(original_image_rgb)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Detected Objects")
    plt.imshow(annotated_image_rgb)
    plt.axis("off")
    legend_handles = []
    for class_name, color in class_labels.items():
        normalized_color = np.array(color) / 255.0
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                         label=class_name, markerfacecolor=normalized_color, markersize=10))
    if legend_handles:
        plt.legend(handles=legend_handles, loc="upper right", title="Classes")
    plt.tight_layout()
    output_dir = os.getenv("OUTPUT_DIR", "./violation_images")
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_filename = f"detected_objects_{timestamp_str}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"[INFO] Annotated result saved to: {output_path}")
    return output_path

def extract_face_embedding(image, face_box):
    x1, y1, x2, y2 = face_box
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
        return 1.0
    similarity = dot / (norm1 * norm2)
    return 1 - similarity

def compare_embeddings(embedding1, embedding2, threshold):
    dist = calculate_cosine_distance(embedding1, embedding2)
    return dist < threshold

# ----------------- External Trigger ----------------- #
async def trigger_external_event(event_url, payload):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(event_url, json=payload, timeout=10)
            return response.status_code, response.text
        except Exception as e:
            logging.error("Error triggering external event: %s", e)
            return None, str(e)
