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
from typing import List, Dict, Optional

# Try to import psycopg2 for Postgres support
try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Database fallback path for SQLite
DB_PATH = os.getenv("SQLITE_DB_PATH", "violations.db")

def is_using_postgres():
    """Returns True if PostgreSQL is configured and available, False for SQLite"""
    database_url = os.getenv("DATABASE_URL")
    return database_url is not None and psycopg2 is not None

def get_sqlite_connection():
    """Returns a direct database connection respecting the configured database type"""
    if is_using_postgres():
        database_url = os.getenv("DATABASE_URL")
        try:
            return psycopg2.connect(database_url)
        except Exception as e:
            logging.error("Error connecting to Postgres: %s", e)
    # Fallback to SQLite
    import sqlite3
    return sqlite3.connect(DB_PATH)

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

# ----------------- Videos Table Operations ----------------- #
def check_camera_exists(tenant_id: str, camera_id: str) -> bool:
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT COUNT(*) FROM videos WHERE tenant_id = ? AND camera_id = ?"
    c.execute(format_query(query, db_type), (tenant_id, camera_id))
    count = c.fetchone()[0]
    conn.close()
    return count > 0

def insert_video_record(tenant_id: str, camera_id: str, is_live: bool, filename: str, stream_url: Optional[str] = None, s3_url: Optional[str] = None) -> str:
    """Insert a new video record into the database"""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    # Check if camera exists for this tenant
    cursor.execute("""
        SELECT camera_id FROM cameras 
        WHERE tenant_id = %s AND camera_id = %s
    """, (tenant_id, camera_id))
    
    if not cursor.fetchone():
        raise ValueError(f"Camera {camera_id} not found for tenant {tenant_id}")
    
    # Check if video already exists for this camera
    cursor.execute("""
        SELECT video_id FROM videos 
        WHERE tenant_id = %s AND camera_id = %s
    """, (tenant_id, camera_id))
    
    existing = cursor.fetchone()
    if existing:
        raise ValueError(f"Video already exists for camera {camera_id}")
    
    # Insert new video record
    cursor.execute("""
        INSERT INTO videos (
            tenant_id, camera_id, is_live, filename, 
            stream_url, s3_url, status, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        RETURNING video_id
    """, (tenant_id, camera_id, is_live, filename, stream_url, s3_url, "pending"))
    
    video_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return video_id

def list_video_records(tenant_id: str) -> List[Dict]:
    """List all video records for a tenant"""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            video_id, camera_id, is_live, filename, 
            stream_url, s3_url, status
        FROM videos 
        WHERE tenant_id = %s
    """, (tenant_id,))
    
    videos = []
    for row in cursor.fetchall():
        videos.append({
            "video_id": row[0],
            "camera_id": row[1],
            "is_live": row[2],
            "filename": row[3],
            "stream_url": row[4],
            "s3_url": row[5],
            "status": row[6]
        })
    
    conn.close()
    return videos

def get_video_record(video_id: str) -> Optional[Dict]:
    """Get a specific video record by video_id"""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            video_id, tenant_id, camera_id, is_live, 
            filename, stream_url, s3_url, status, 
            created_at, updated_at
        FROM videos 
        WHERE video_id = %s
    """, (video_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "video_id": row[0],
        "tenant_id": row[1],
        "camera_id": row[2],
        "is_live": row[3],
        "filename": row[4],
        "stream_url": row[5],
        "s3_url": row[6],
        "status": row[7],
        "created_at": row[8].isoformat() if row[8] else None,
        "updated_at": row[9].isoformat() if row[9] else None
    }

def get_video_record_by_camera(tenant_id: str, camera_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT video_id, tenant_id, camera_id, is_live, filename, stream_url, status, size, fps, total_frames, duration, frames_processed, violations_detected FROM videos WHERE tenant_id = ? AND camera_id = ?"
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

def update_video_record(video_id: str, tenant_id: str, camera_id: str, is_live: Optional[bool] = None, 
                       stream_url: Optional[str] = None, status: Optional[str] = None, 
                       filename: Optional[str] = None, s3_url: Optional[str] = None) -> bool:
    """Update an existing video record"""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    # Check if video exists
    cursor.execute("""
        SELECT video_id FROM videos 
        WHERE video_id = %s AND tenant_id = %s AND camera_id = %s
    """, (video_id, tenant_id, camera_id))
    
    if not cursor.fetchone():
        conn.close()
        return False
    
    # Build update query dynamically
    update_fields = []
    update_values = []
    
    if is_live is not None:
        update_fields.append("is_live = %s")
        update_values.append(is_live)
    if stream_url is not None:
        update_fields.append("stream_url = %s")
        update_values.append(stream_url)
    if status is not None:
        update_fields.append("status = %s")
        update_values.append(status)
    if filename is not None:
        update_fields.append("filename = %s")
        update_values.append(filename)
    if s3_url is not None:
        update_fields.append("s3_url = %s")
        update_values.append(s3_url)
    
    if update_fields:
        update_fields.append("updated_at = NOW()")
        update_values.extend([video_id, tenant_id, camera_id])
        
        query = f"""
            UPDATE videos 
            SET {", ".join(update_fields)}
            WHERE video_id = %s AND tenant_id = %s AND camera_id = %s
        """
        
        cursor.execute(query, update_values)
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

def delete_video_record_by_camera(tenant_id: str, camera_id: str) -> bool:
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT filename FROM videos WHERE tenant_id = ? AND camera_id = ?"
    c.execute(format_query(query, db_type), (tenant_id, camera_id))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    filename = row[0]
    query = "DELETE FROM videos WHERE tenant_id = ? AND camera_id = ?"
    c.execute(format_query(query, db_type), (tenant_id, camera_id))
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

def update_video_record_by_camera(tenant_id: str, camera_id: str, fields: dict) -> bool:
    if not fields:
        return False
    conn, db_type = get_connection()
    c = conn.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in fields.keys()])
    values = list(fields.values())
    values.extend([tenant_id, camera_id])
    query = f"UPDATE videos SET {set_clause} WHERE tenant_id = ? AND camera_id = ?"
    c.execute(format_query(query, db_type), tuple(values))
    conn.commit()
    conn.close()

# ----------------- Tenant Config Operations ----------------- #
def get_tenant_config(tenant_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT tenant_name, similarity_threshold, mask_threshold_minutes, vest_threshold_minutes, hardhat_threshold_minutes, external_trigger_url, is_active FROM tenant_config WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (tenant_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "tenant_name": row[0],
            "similarity_threshold": row[1],
            "no_mask_threshold": row[2],
            "no_safety_vest_threshold": row[3],
            "no_hardhat_threshold": row[4],
            "external_trigger_url": row[5],
            "is_active": bool(row[6])
        }
    return None

def add_or_update_tenant_config(tenant_id, tenant_name, similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url, is_active=True):
    existing = get_tenant_config(tenant_id)
    conn, db_type = get_connection()
    c = conn.cursor()
    if existing:
        query = "UPDATE tenant_config SET tenant_name = ?, similarity_threshold = ?, mask_threshold_minutes = ?, vest_threshold_minutes = ?, hardhat_threshold_minutes = ?, external_trigger_url = ?, is_active = ? WHERE tenant_id = ?"
        c.execute(format_query(query, db_type), (tenant_name, similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url, is_active, tenant_id))
    else:
        query = "INSERT INTO tenant_config (tenant_id, tenant_name, similarity_threshold, mask_threshold_minutes, vest_threshold_minutes, hardhat_threshold_minutes, external_trigger_url, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        c.execute(format_query(query, db_type), (tenant_id, tenant_name, similarity_threshold, no_mask_threshold, no_safety_vest_threshold, no_hardhat_threshold, external_trigger_url, is_active))
    conn.commit()
    conn.close()

def delete_tenant_config(tenant_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "DELETE FROM tenant_config WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (tenant_id,))
    conn.commit()
    conn.close()

def list_tenants():
    """
    Lists all tenants with their configurations, active status, and associated cameras.
    Returns a list of dictionaries containing tenant information.
    """
    conn, db_type = get_connection()
    c = conn.cursor()
    try:
        # Get all tenants with their configurations
        query = """
        SELECT 
            tc.tenant_id,
            tc.similarity_threshold,
            tc.mask_threshold_minutes,
            tc.vest_threshold_minutes,
            tc.hardhat_threshold_minutes,
            tc.external_trigger_url,
            tc.is_active,
            COUNT(DISTINCT v.camera_id) as total_cameras,
            COUNT(DISTINCT f.face_id) as total_faces
        FROM tenant_config tc
        LEFT JOIN videos v ON tc.tenant_id = v.tenant_id
        LEFT JOIN faces f ON tc.tenant_id = f.tenant_id
        GROUP BY tc.tenant_id
        """
        c.execute(format_query(query, db_type))
        tenants = []
        
        for row in c.fetchall():
            tenant_id = row[0]
            # Get cameras for this tenant
            camera_query = """
            SELECT 
                video_id,
                camera_id,
                is_live,
                filename,
                stream_url,
                status,
                size,
                fps,
                total_frames,
                duration,
                frames_processed,
                violations_detected
            FROM videos 
            WHERE tenant_id = ?
            """
            c.execute(format_query(camera_query, db_type), (tenant_id,))
            cameras = []
            for cam_row in c.fetchall():
                cameras.append({
                    "video_id": cam_row[0],
                    "camera_id": cam_row[1],
                    "is_live": bool(cam_row[2]),
                    "filename": cam_row[3],
                    "stream_url": cam_row[4],
                    "status": cam_row[5],
                    "size": cam_row[6],
                    "fps": cam_row[7],
                    "total_frames": cam_row[8],
                    "duration": cam_row[9],
                    "frames_processed": cam_row[10],
                    "violations_detected": cam_row[11]
                })
            
            tenants.append({
                "tenant_id": tenant_id,
                "config": {
                    "similarity_threshold": row[1],
                    "no_mask_threshold": row[2],
                    "no_safety_vest_threshold": row[3],
                    "no_hardhat_threshold": row[4],
                    "external_trigger_url": row[5],
                    "is_active": bool(row[6])
                },
                "total_cameras": row[7],
                "total_faces": row[8],
                "cameras": cameras
            })
            
        return tenants
    finally:
        conn.close()

def update_tenant_status(tenant_id: str, is_active: bool):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "UPDATE tenant_config SET is_active = ? WHERE tenant_id = ?"
    c.execute(format_query(query, db_type), (is_active, tenant_id))
    conn.commit()
    conn.close()

# ----------------- Faces Table Operations ----------------- #
def add_face_record(tenant_id: str, camera_id: str, face_id: str, name: str, embedding: str, metadata: str = None, image_path: str = None, s3_url: str = None) -> str:
    """
    Adds a new face record to the database.
    Validates that the camera exists for the tenant before adding the face.
    Returns the face_id if successful.
    """
    conn, db_type = get_connection()
    c = conn.cursor()
    try:
        # First verify that the camera exists for this tenant
        if not check_camera_exists(tenant_id, camera_id):
            raise ValueError(f"Camera {camera_id} does not exist for tenant {tenant_id}")

        # Check if face_id already exists
        c.execute(format_query("SELECT face_id FROM faces WHERE face_id = ?", db_type), (face_id,))
        if c.fetchone():
            raise ValueError(f"Face ID {face_id} already exists")

        # Insert the new face record
        query = """
        INSERT INTO faces (face_id, tenant_id, camera_id, name, embedding, metadata, image_path, s3_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        c.execute(format_query(query, db_type), (face_id, tenant_id, camera_id, name, embedding, metadata, image_path, s3_url))
        conn.commit()
        return face_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def update_face_record(face_id: str, tenant_id: str, camera_id: str, name: str = None, embedding: str = None, metadata: str = None, image_path: str = None, s3_url: str = None) -> bool:
    """
    Updates an existing face record in the database.
    Validates that the camera exists for the tenant before updating the face.
    Returns True if successful.
    """
    conn, db_type = get_connection()
    c = conn.cursor()
    try:
        # First verify that the camera exists for this tenant
        if not check_camera_exists(tenant_id, camera_id):
            raise ValueError(f"Camera {camera_id} does not exist for tenant {tenant_id}")

        # Check if face exists
        c.execute(format_query("SELECT face_id FROM faces WHERE face_id = ?", db_type), (face_id,))
        if not c.fetchone():
            raise ValueError(f"Face ID {face_id} does not exist")

        # Build update query dynamically based on provided fields
        update_fields = []
        params = []
        if name is not None:
            update_fields.append("name = ?")
            params.append(name)
        if embedding is not None:
            update_fields.append("embedding = ?")
            params.append(embedding)
        if metadata is not None:
            update_fields.append("metadata = ?")
            params.append(metadata)
        if image_path is not None:
            update_fields.append("image_path = ?")
            params.append(image_path)
        if s3_url is not None:
            update_fields.append("s3_url = ?")
            params.append(s3_url)
        
        if not update_fields:
            return True  # No fields to update

        # Add tenant_id and camera_id to params
        params.extend([tenant_id, camera_id, face_id])
        
        query = f"""
        UPDATE faces 
        SET {', '.join(update_fields)}
        WHERE tenant_id = ? AND camera_id = ? AND face_id = ?
        """
        
        c.execute(format_query(query, db_type), params)
        conn.commit()
        
        if c.rowcount == 0:
            raise ValueError(f"No face found with ID {face_id} for tenant {tenant_id} and camera {camera_id}")
            
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
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
    query = "SELECT face_id, camera_id, name, embedding, metadata, image_path, s3_url FROM faces WHERE tenant_id = ?"
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
            "metadata": json.loads(row[4]) if row[4] else None,
            "image_path": row[5],
            "s3_url": row[6]
        })
    return faces

def get_face_record(face_id: str):
    conn, db_type = get_connection()
    c = conn.cursor()
    query = "SELECT face_id, tenant_id, camera_id, name, embedding, metadata, image_path, s3_url FROM faces WHERE face_id = ?"
    c.execute(format_query(query, db_type), (face_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "face_id": row[0],
            "tenant_id": row[1],
            "camera_id": row[2],
            "name": row[3],
            "embedding": json.loads(row[4]),
            "metadata": json.loads(row[5]) if row[5] else None,
            "image_path": row[6],
            "s3_url": row[7]
        }
    return None

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
    output_filename = f"violation_{timestamp_str}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # Just save the annotated image directly
    try:
        # Convert from BGR to RGB if necessary
        if isinstance(annotated_image_rgb, np.ndarray) and annotated_image_rgb.shape[2] == 3:
            # Check if image is already in BGR (OpenCV format)
            if np.array_equal(annotated_image_rgb[:, :, 0], original_bgr_image[:, :, 0]):
                save_img = annotated_image_rgb
            else:
                save_img = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
        else:
            save_img = original_bgr_image
            
        cv2.imwrite(output_path, save_img)
        logging.info(f"[INFO] Violation image saved to: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"[ERROR] Failed to save violation image: {str(e)}")
        # Fallback to saving the original image
        try:
            cv2.imwrite(output_path, original_bgr_image)
            logging.info(f"[INFO] Fallback: Original image saved to: {output_path}")
            return output_path
        except Exception as e2:
            logging.error(f"[ERROR] Also failed to save original image: {str(e2)}")
            return None

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

def ensure_tables_exist():
    """Make sure the required database tables exist"""
    conn, db_type = get_connection()
    c = conn.cursor()
    
    try:
        # Create violations table if it doesn't exist
        if db_type == "postgres":
            c.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                violation_timestamp REAL NOT NULL,
                face_id TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_image_path TEXT,
                details TEXT
            )
            """)
        else:  # sqlite
            c.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                violation_timestamp REAL NOT NULL,
                face_id TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_image_path TEXT,
                details TEXT
            )
            """)
            
        # Create videos table if it doesn't exist
        if db_type == "postgres":
            c.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                is_live INTEGER NOT NULL DEFAULT 0,
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
            """)
        else:  # sqlite
            c.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                is_live INTEGER NOT NULL DEFAULT 0,
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
            """)
            
        # Create other tables as needed (faces, tenant_config, etc.)
        
        conn.commit()
    except Exception as e:
        logging.error(f"Error ensuring tables exist: {str(e)}")
    finally:
        conn.close()

def get_video_statistics(tenant_id: str) -> Dict:
    """Get video statistics for a specific tenant"""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            video_id,
            tenant_id,
            camera_id,
            is_live,
            duration,
            status,
            frames_processed,
            violations_detected
        FROM videos 
        WHERE tenant_id = %s
    """, (tenant_id,))
    
    total_duration = 0
    violation_counts = {}
    processing_videos = []
    
    for row in cursor.fetchall():
        video_id = row[0]
        duration = row[4] or 0
        status = row[5]
        
        total_duration += duration
        violation_counts[video_id] = row[7] or 0
        
        if status == "processing":
            processing_videos.append({
                "video_id": video_id,
                "tenant_id": row[1],
                "camera_id": row[2],
                "is_live": bool(row[3]),
                "frames_processed": row[6] or 0,
                "violations_detected": row[7] or 0
            })
    
    conn.close()
    return {
        "total_duration": total_duration,
        "violation_counts": violation_counts,
        "processing_videos": processing_videos
    }
