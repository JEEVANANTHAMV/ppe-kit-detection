import sqlite3
import json
import os
import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import httpx
from deepface import DeepFace
from ultralytics import YOLO

DB_PATH = "violations.db"

# 1) YOLO model initialization
MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH)

# 2) Required items
REQUIRED_ITEMS = {"No-Mask", "No-Safety-Vest", "No-Hardhat"}

############################################################################
# Database Initialization
############################################################################

def init_db():
    """
    Initializes the database and creates required tables if not exist:
      - tenant_config
      - videos
      - faces
      - violations
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Tenant config
    c.execute("""
        CREATE TABLE IF NOT EXISTS tenant_config (
            tenant_id TEXT PRIMARY KEY,
            similarity_threshold REAL,
            violation_threshold INTEGER,
            external_trigger_url TEXT
        )
    """)
    # Videos
    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """)
    # Faces
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            name TEXT,
            embedding TEXT,
            metadata TEXT
        )
    """)
    # Violations
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT,
            camera_id TEXT,
            violation_timestamp REAL,
            face_id INTEGER,
            violation_type TEXT,
            violation_image_path TEXT,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()


############################################################################
# Videos Table Operations
############################################################################

def check_camera_exists(tenant_id: str, camera_id: str) -> bool:
    """
    Returns True if there's a video/stream with the same (tenant_id, camera_id).
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM videos WHERE tenant_id = ? AND camera_id = ?", (tenant_id, camera_id))
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
    """
    Inserts a new row in the 'videos' table and returns the generated video_id.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO videos
        (tenant_id, camera_id, is_live, filename, stream_url, size, fps, total_frames, duration, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tenant_id,
        camera_id,
        1 if is_live else 0,
        filename,
        stream_url,
        size,
        fps,
        total_frames,
        duration,
        "registered" if is_live else "uploaded"
    ))
    conn.commit()
    video_id = c.lastrowid
    conn.close()
    return video_id


############################################################################
# Tenant Config Table Operations
############################################################################

def get_tenant_config(tenant_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT similarity_threshold, violation_threshold, external_trigger_url
        FROM tenant_config
        WHERE tenant_id = ?
    """, (tenant_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "similarity_threshold": row[0],
            "violation_threshold": row[1],
            "external_trigger_url": row[2]
        }
    return None


def add_or_update_tenant_config(tenant_id, similarity_threshold, violation_threshold, external_trigger_url):
    existing = get_tenant_config(tenant_id)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if existing:
        c.execute("""
            UPDATE tenant_config
            SET similarity_threshold = ?, violation_threshold = ?, external_trigger_url = ?
            WHERE tenant_id = ?
        """, (similarity_threshold, violation_threshold, external_trigger_url, tenant_id))
    else:
        c.execute("""
            INSERT INTO tenant_config (tenant_id, similarity_threshold, violation_threshold, external_trigger_url)
            VALUES (?, ?, ?, ?)
        """, (tenant_id, similarity_threshold, violation_threshold, external_trigger_url))
    conn.commit()
    conn.close()


def delete_tenant_config(tenant_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM tenant_config WHERE tenant_id = ?", (tenant_id,))
    conn.commit()
    conn.close()


############################################################################
# Faces Table Operations
############################################################################

def add_face_record(tenant_id, camera_id, name, embedding, metadata) -> int:
    """
    Inserts a new face record.
    Embedding is stored as JSON string, so it can be reloaded as a list/np.array.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    embedding_json = json.dumps(embedding)
    metadata_json = json.dumps(metadata)
    c.execute("""
        INSERT INTO faces (tenant_id, camera_id, name, embedding, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (tenant_id, camera_id, name, embedding_json, metadata_json))
    conn.commit()
    face_id = c.lastrowid
    conn.close()
    return face_id


def update_face_record(face_id, tenant_id, camera_id, name, embedding, metadata):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    embedding_json = json.dumps(embedding)
    metadata_json = json.dumps(metadata)
    c.execute("""
        UPDATE faces
        SET camera_id = ?, name = ?, embedding = ?, metadata = ?
        WHERE face_id = ? AND tenant_id = ?
    """, (camera_id, name, embedding_json, metadata_json, face_id, tenant_id))
    conn.commit()
    conn.close()


def delete_face_record(face_id, tenant_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM faces WHERE face_id = ? AND tenant_id = ?", (face_id, tenant_id))
    conn.commit()
    conn.close()


def list_face_records(tenant_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT face_id, camera_id, name, embedding, metadata
        FROM faces
        WHERE tenant_id = ?
    """, (tenant_id,))
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


############################################################################
# Violations Table Operations
############################################################################

def save_violation_to_db(
    tenant_id,
    camera_id,
    violation_timestamp,
    face_id,
    violation_type,
    violation_image_path,
    details=""
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO violations
        (tenant_id, camera_id, violation_timestamp, face_id, violation_type, violation_image_path, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
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


############################################################################
# Detection & Recognition
############################################################################

def detect_objects(image):
    """
    YOLO detection.
    Returns:
      - boxes
      - results
      - annotated_image (RGB)
      - colors
      - class_confidences
      - violation_faces: list of (person_id, (x1, y1, x2, y2)) who is missing items
    """
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

        if class_name not in class_confidences:
            class_confidences[class_name] = []
        class_confidences[class_name].append(conf)

        if class_name == "Person":
            person_id = f"Person_{i}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check if person is missing any required item
            detected_items = set()
            for j, sub_box in enumerate(boxes):
                sub_cls_id = int(sub_box.cls[0])
                sub_class_name = results.names[sub_cls_id]
                sx1, sy1, sx2, sy2 = map(int, sub_box.xyxy[0])
                # If sub_box is within person's bounding box, assume that item belongs to this person
                if x1 <= sx1 <= x2 and y1 <= sy1 <= y2:
                    detected_items.add(sub_class_name)

            missing_items = REQUIRED_ITEMS - detected_items
            if missing_items:
                # Mark this person as violating
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_image, "Violation", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                violation_faces.append((person_id, (x1, y1, x2, y2)))

    return boxes, results, annotated_image, colors, class_confidences, violation_faces


def check_violations(class_confidences, threshold=0.5):
    """
    Identifies any "NO-*" classes that exceed the confidence threshold.
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
    Saves a side-by-side comparison of the original and annotated frames.
    Returns the output path to the saved image.
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
        legend_handles.append(
            plt.Line2D([0], [0],
                       marker="o",
                       color="w",
                       label=class_name,
                       markerfacecolor=normalized_color,
                       markersize=10)
        )
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

    print(f"[INFO] Annotated result saved to: {output_path}")
    return output_path


############################################################################
# Face Embedding & Comparison
############################################################################

def extract_face_embedding(image, face_box):
    """
    Extracts a face from the image using (x1, y1, x2, y2) and returns its embedding via DeepFace.
    If no face is found or enforce_detection fails, returns None.
    """
    (x1, y1, x2, y2) = face_box
    face_crop = image[y1:y2, x1:x2]
    try:
        embeddings = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)
        if isinstance(embeddings, list) and len(embeddings) > 0:
            return embeddings[0]["embedding"]
    except Exception as e:
        print("extract_face_embedding error:", e)
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
    distance = 1 - similarity
    return distance


def compare_embeddings(embedding1, embedding2, threshold):
    """
    Returns True if the cosine distance between embedding1 & embedding2 is < threshold.
    """
    dist = calculate_cosine_distance(embedding1, embedding2)
    return dist < threshold


############################################################################
# External Trigger
############################################################################

async def trigger_external_event(event_url, payload):
    """
    Fires a POST request to the specified event_url with JSON payload.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(event_url, json=payload, timeout=10)
            return response.status_code, response.text
        except Exception as e:
            print(f"Error triggering external event: {e}")
            return None, str(e)
