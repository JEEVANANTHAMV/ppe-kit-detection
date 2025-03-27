import os
import cv2
import time
import json
import logging
import torch
import asyncio
from multiprocessing import Process, Queue
from deepface import DeepFace
from ultralytics import YOLO
import faiss
import numpy as np
from helper import (
    get_video_record,
    get_tenant_config,
    increment_violations_detected,
    increment_frames_processed,
    save_annotated_plot,
    save_violation_to_db,
    trigger_external_event,
    find_matching_faces,
    list_face_records,
    update_video_status,
    DB_PATH,
    get_sqlite_connection,
    ensure_tables_exist,
    is_using_postgres
)
import requests
import datetime
import uuid
import traceback
import psycopg2

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Create necessary directories
os.makedirs("violation_images", exist_ok=True)

# Get configurable thresholds from environment variables
PERSON_CONFIDENCE_THRESHOLD = float(os.getenv("PERSON_CONFIDENCE_THRESHOLD", "0.8"))
EQUIPMENT_CONFIDENCE_THRESHOLD = float(os.getenv("EQUIPMENT_CONFIDENCE_THRESHOLD", "0.8"))
VIOLATION_CONFIDENCE_THRESHOLD = float(os.getenv("VIOLATION_CONFIDENCE_THRESHOLD", "0.8"))

def get_threshold_for_violation(config, violation_type):
    threshold_mapping = {
        "NO-Mask": config.get("mask_threshold_minutes", 5),
        "NO-Safety-Vest": config.get("vest_threshold_minutes", 5),
        "NO-Hardhat": config.get("hardhat_threshold_minutes", 5)
    }
    return threshold_mapping.get(violation_type, 5)  # Default to 5 minutes if not specified

class VideoProcessor:
    def __init__(self, model_path="weights/best.pt"):
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.gpu_semaphore = asyncio.Semaphore(4)  # Limit concurrent GPU operations
        
    async def process_frame(self, frame, tenant_id, camera_id, similarity_threshold, violation_timers, violation_timers_lock, video_record):
        try:
            async with self.gpu_semaphore:
                boxes, results, annotated_img, colors, class_confidences, violation_faces = await self.detect_objects(frame)
                violations_found = self.check_violations(class_confidences)
                
                if violations_found:
                    # Process violations
                    current_time = time.time()
                    config = get_tenant_config(tenant_id)
                    
                    for person_id, (x1, y1, x2, y2) in violation_faces:
                        embedding = self.extract_face_embedding(frame, (x1, y1, x2, y2))
                        face_id = "unknown_face"
                        
                        if embedding is not None:
                            matched_face_ids = self.find_matching_faces(tenant_id, embedding, similarity_threshold)
                            if matched_face_ids:
                                face_id = matched_face_ids[0]
                        
                        for vio in violations_found:
                            key = (tenant_id, camera_id, face_id, vio)
                            threshold_minutes = get_threshold_for_violation(config, vio)
                            threshold_seconds = threshold_minutes * 60
                            
                            async with violation_timers_lock:
                                if key not in violation_timers:
                                    violation_timers[key] = current_time
                                else:
                                    elapsed = current_time - violation_timers[key]
                                    if elapsed >= threshold_seconds:
                                        # Save annotated image
                                        annotated_path = save_annotated_plot(
                                            original_bgr_image=frame,
                                            boxes=boxes,
                                            class_names=results.names,
                                            annotated_image_rgb=annotated_img,
                                            colors=colors,
                                            confidence_threshold=0.5,
                                            video_source=video_record["stream_url"] if video_record["is_live"] else video_record["filename"]
                                        )
                                        
                                        # Save violation to database
                                        violation_record = {
                                            "tenant_id": tenant_id,
                                            "camera_id": camera_id,
                                            "violation_timestamp": current_time,
                                            "face_id": face_id,
                                            "violation_type": vio,
                                            "violation_image_path": annotated_path,
                                            "details": json.dumps({
                                                "person_id": person_id,
                                                "elapsed_seconds": elapsed,
                                                "face_detected": embedding is not None
                                            })
                                        }
                                        save_violation_to_db(**violation_record)
                                        
                                        # Update violation timer and increment count
                                        violation_timers[key] = current_time
                                        increment_violations_detected(video_record["video_id"])
                                        
                                        # Trigger external event if configured
                                        if config.get("external_trigger_url"):
                                            try:
                                                asyncio.create_task(trigger_external_event(
                                                    config["external_trigger_url"], 
                                                    violation_record
                                                ))
                                            except Exception as e:
                                                logging.error(f"Failed to trigger external event: {str(e)}")
                    
                    # Clean up violation timers
                    async with violation_timers_lock:
                        for key in list(violation_timers.keys()):
                            if key[0] == tenant_id and key[1] == camera_id and key[2] == face_id and key[3] not in violations_found:
                                violation_timers.pop(key, None)
                
                return annotated_img, violations_found
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            return None, []

    async def detect_objects(self, frame):
        results = self.model(frame, conf=0.5)
        boxes = []
        class_confidences = {}
        violation_faces = []
        colors = {}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = str(r.names[cls])  # Ensure class_name is a string
                class_confidences[class_name] = conf
                
                if class_name == "person" and conf > PERSON_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    violation_faces.append((len(violation_faces), (x1, y1, x2, y2)))
                
                colors[class_name] = (0, 0, 255) if class_name in ["NO-Mask", "NO-Safety-Vest", "NO-Hardhat"] else (0, 255, 0)
        
        # Create annotated image manually
        annotated_img = frame.copy()
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = str(r.names[cls])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw box and label
            color = colors[class_name]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return boxes, results[0], annotated_img, colors, class_confidences, violation_faces

    def check_violations(self, class_confidences, threshold=None):
        violations = []
        
        # Use provided threshold or default to the environment variable
        if threshold is None:
            threshold = VIOLATION_CONFIDENCE_THRESHOLD
        
        # Handle both formats of class_confidences (dict of lists or dict of values)
        for violation in ["NO-Mask", "NO-Safety-Vest", "NO-Hardhat"]:
            if violation in class_confidences:
                if isinstance(class_confidences[violation], list):
                    # Format from detect_objects_sync: list of confidences
                    if any(conf > threshold for conf in class_confidences[violation]):
                        violations.append(violation)
                else:
                    # Format from other methods: single confidence value
                    if class_confidences[violation] > threshold:
                        violations.append(violation)
        
        logging.info(f"Violations found: {violations} (threshold: {threshold})")
        return violations

    def extract_face_embedding(self, frame, bbox):
        try:
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]
            embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
            return embedding[0]["embedding"] if embedding else None
        except Exception as e:
            logging.error(f"Error extracting face embedding: {str(e)}")
            return None

    def find_matching_faces(self, tenant_id, embedding, similarity_threshold):
        try:
            # Get all faces for the tenant
            faces = list_face_records(tenant_id)
            if not faces:
                return None
                
            # Convert embeddings to numpy array
            embeddings = []
            face_ids = []
            for face in faces:
                if face.get("embedding"):
                    try:
                        face_embedding = json.loads(face["embedding"])
                        embeddings.append(face_embedding)
                        face_ids.append(face["face_id"])
                    except:
                        continue
            
            if not embeddings:
                return None
                
            embeddings = np.array(embeddings)
            
            # Create FAISS index
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            # Search for matches
            D, I = index.search(np.array([embedding]).astype('float32'), k=1)
            
            # Check similarity threshold
            if D[0][0] < similarity_threshold:
                return [face_ids[I[0][0]]]
            return None
            
        except Exception as e:
            logging.error(f"Error in face matching: {str(e)}")
            return None

    def handle_violation(self, tenant_id, camera_id, face_id, violation_type, timestamp, elapsed, face_detected):
        # Implementation of violation handling logic
        pass

    def detect_objects_sync(self, frame):
        """A synchronous version of detect_objects for use in multiprocessing"""
        results = self.model(frame, conf=0.5)
        boxes = []
        class_confidences = {}
        violation_faces = []
        colors = {}
        
        # Detect if any safety equipment is missing
        safety_equipment = {"Hardhat": False, "Mask": False, "Safety-Vest": False}
        
        # Track if we found a person with high confidence
        person_detected = False
        person_confidence = 0.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = str(r.names[cls])  # Ensure class_name is a string
                
                # Log all class names to debug
                logging.info(f"Detected class: {class_name} with confidence {conf}")
                
                # Track safety equipment
                if class_name == "Hardhat" and conf > EQUIPMENT_CONFIDENCE_THRESHOLD:
                    safety_equipment["Hardhat"] = True
                elif class_name == "Mask" and conf > EQUIPMENT_CONFIDENCE_THRESHOLD:
                    safety_equipment["Mask"] = True
                elif (class_name == "Vest" or class_name == "Safety Vest" or class_name == "Safety-Vest") and conf > EQUIPMENT_CONFIDENCE_THRESHOLD:
                    safety_equipment["Safety-Vest"] = True
                
                # Check for person using various possible class names
                if class_name.lower() in ["person", "Person"]:
                    person_confidence = conf
                    if conf > PERSON_CONFIDENCE_THRESHOLD:
                        person_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        violation_faces.append((len(violation_faces), (x1, y1, x2, y2)))
                        logging.info(f"Found person at ({x1}, {y1}, {x2}, {y2}) with confidence {conf}")
                
                # Create a dict of class names to confidence values for original detections
                class_confidences.setdefault(class_name, []).append(conf)
        
        # Only add violation entries if a person was detected with high confidence
        if person_detected:
            # Create violation entries for missing equipment
            if not safety_equipment["Hardhat"]:
                class_confidences.setdefault("NO-Hardhat", []).append(0.9)
            
            if not safety_equipment["Mask"]:
                class_confidences.setdefault("NO-Mask", []).append(0.9)
                
            if not safety_equipment["Safety-Vest"]:
                class_confidences.setdefault("NO-Safety-Vest", []).append(0.9)
        else:
            # If no person detected with high confidence, log and return
            logging.info(f"No person detected with high confidence (max: {person_confidence:.2f}, threshold: {PERSON_CONFIDENCE_THRESHOLD})")
            # Clear any violation entries that might have been added
            for violation in ["NO-Hardhat", "NO-Mask", "NO-Safety-Vest"]:
                if violation in class_confidences:
                    del class_confidences[violation]
            
            # If we don't have a valid person detection, don't create violation_faces
            violation_faces = []
        
        # Define colors
        colors = {
            "NO-Hardhat": (0, 0, 255),   # Red
            "NO-Mask": (0, 0, 255),      # Red
            "NO-Safety-Vest": (0, 0, 255), # Red
            "Hardhat": (0, 255, 0),      # Green
            "Mask": (0, 255, 0),         # Green
            "Safety-Vest": (0, 255, 0)   # Green
        }
        
        # Create annotated image manually
        annotated_img = frame.copy()
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = str(r.names[cls])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw box and label
            color = colors.get(class_name, (0, 255, 0))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add text for missing equipment
        y_offset = 30
        for item, present in safety_equipment.items():
            if not present:
                label = f"NO-{item}"
                cv2.putText(annotated_img, label, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        # Print detected objects and violations for debugging
        logging.info(f"Detected objects: {list(class_confidences.keys())}")
        logging.info(f"Safety equipment: {safety_equipment}")
        logging.info(f"Found {len(violation_faces)} potential faces to check")
        
        return boxes, results[0], annotated_img, colors, class_confidences, violation_faces

def process_video(video_id, tenant_id, camera_id, similarity_threshold, violation_timers, violation_timers_lock, status_queue):
    processor = VideoProcessor()
    cap = None
    
    # Ensure database tables exist first
    try:
        ensure_tables_exist()
    except Exception as e:
        logging.error(f"[ERROR] Failed to ensure database tables exist: {str(e)}")
        status_queue.put(("error", f"Database initialization failed: {str(e)}"))
        update_video_status(video_id, "error")
        return
    
    # Create a database check to see if this employee+violation has been reported before
    conn = get_sqlite_connection()
    c = conn.cursor()
    
    # Load previously detected violations for this camera
    previously_detected_violations = set()
    try:
        is_postgres = is_using_postgres()
        if is_postgres:
            query = """
            SELECT DISTINCT face_id, violation_type 
            FROM violations 
            WHERE tenant_id = %s AND camera_id = %s
            """
            c.execute(query, (tenant_id, camera_id))
        else:
            query = """
            SELECT DISTINCT face_id, violation_type 
            FROM violations 
            WHERE tenant_id = ? AND camera_id = ?
            """
            c.execute(query, (tenant_id, camera_id))
            
        for row in c.fetchall():
            face_id, violation_type = row
            key = (tenant_id, camera_id, face_id, violation_type)
            previously_detected_violations.add(key)
            logging.info(f"Found previously detected violation: {key}")
    except Exception as e:
        logging.error(f"[ERROR] Error loading previous violations: {str(e)}")
    finally:
        conn.close()
    
    # Local violation tracker for this process
    local_violation_timers = {}
    # Track first-time violations for each employee, initialized with previously detected ones
    first_time_violations = previously_detected_violations
    
    update_video_status(video_id, "processing")
    logging.info(f"Starting processing for video {video_id}")
    
    try:
        # Initialize video capture
        video_record = get_video_record(video_id)
        if not video_record:
            status_queue.put(("error", f"Video record not found for video_id {video_id}"))
            update_video_status(video_id, "error")
            return

        cap_source = video_record["stream_url"] if video_record["is_live"] else video_record["filename"]
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            status_queue.put(("error", f"Cannot open source for video_id {video_id}"))
            update_video_status(video_id, "error")
            return

        # Process frames
        frame_count = 0
        target_fps = 15 if torch.cuda.is_available() else 1
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip_interval = max(1, int(original_fps // target_fps))
        
        # Get tenant config
        config = get_tenant_config(tenant_id)
        if not config:
            status_queue.put(("error", f"No tenant configuration for tenant {tenant_id}"))
            update_video_status(video_id, "no_config")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip_interval != 0:
                continue

            # Process frame directly without asyncio
            try:
                # Detect objects
                boxes, results, annotated_img, colors, class_confidences, violation_faces = processor.detect_objects_sync(frame)
                violations_found = processor.check_violations(class_confidences)
                
                # Only process if we found both violations and faces with high confidence
                if violations_found and violation_faces:
                    # Process violations
                    current_time = time.time()
                    logging.info(f"Processing violations: {violations_found} for {len(violation_faces)} faces")
                    
                    for person_id, (x1, y1, x2, y2) in violation_faces:
                        try:
                            # Extract face embedding
                            face_img = frame[y1:y2, x1:x2]
                            embedding_result = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
                            embedding = embedding_result[0]["embedding"] if embedding_result else None
                            face_id = "unknown_face"
                            
                            if embedding is not None:
                                matched_face_ids = find_matching_faces(tenant_id, embedding, similarity_threshold)
                                if matched_face_ids:
                                    face_id = matched_face_ids[0]
                            
                            # Process each violation type
                            for vio in violations_found:
                                key = (tenant_id, camera_id, face_id, vio)
                                
                                # Record timestamp of when this violation was first seen
                                if key not in first_time_violations:
                                    # First time violation for this employee/violation type - report immediately
                                    logging.info(f"First time violation detected for {key}, reporting immediately")
                                    first_time_violations.add(key)
                                    should_report = True
                                    # Don't start timing yet - this is the first report
                                    local_violation_timers[key] = current_time
                                else:
                                    # This is a repeat violation, check threshold
                                    threshold_minutes = get_threshold_for_violation(config, vio)
                                    threshold_seconds = threshold_minutes * 60
                                    
                                    if key not in local_violation_timers:
                                        # This employee/violation was seen before in a previous run, but not in this session
                                        # Start timer now but don't report yet
                                        local_violation_timers[key] = current_time
                                        should_report = False
                                        logging.info(f"Starting timer for {key}")
                                    else:
                                        # We've seen this violation before and have a timer running
                                        elapsed = current_time - local_violation_timers[key]
                                        should_report = elapsed >= threshold_seconds
                                        logging.info(f"Checking threshold for {key}: {elapsed}/{threshold_seconds} seconds, should report: {should_report}")
                                        
                                        # Reset timer if we're reporting
                                        if should_report:
                                            local_violation_timers[key] = current_time
                                
                                if should_report:
                                    logging.info(f"Recording violation for {key}")
                                    
                                    # Create output directory
                                    if not os.path.exists("violation_images"):
                                        os.makedirs("violation_images", exist_ok=True)
                                        
                                    # Save image directly
                                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    output_filename = f"violation_{timestamp_str}.jpg"
                                    output_path = os.path.join("violation_images", output_filename)
                                    
                                    try:
                                        # Convert to BGR before writing (just to be safe)
                                        if isinstance(annotated_img, np.ndarray):
                                            save_img = annotated_img
                                        else:
                                            save_img = np.array(annotated_img)
                                            
                                        cv2.imwrite(output_path, save_img)
                                        logging.info(f"Violation image saved to: {output_path}")
                                    except Exception as img_error:
                                        logging.error(f"Error saving image: {str(img_error)}")
                                        output_path = None
                                    
                                    if output_path:
                                        # Save violation to database using a direct connection
                                        try:
                                            # Get database connection that respects the configured type
                                            conn = get_sqlite_connection()
                                            cursor = conn.cursor()
                                            
                                            violation_id = str(uuid.uuid4())
                                            
                                            # Check if we're using Postgres or SQLite
                                            is_postgres = is_using_postgres()
                                            
                                            # Insert directly with placeholders appropriate for the database type
                                            if is_postgres:
                                                sql = """
                                                INSERT INTO violations (id, tenant_id, camera_id, violation_timestamp, face_id, violation_type, violation_image_path, details)
                                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                                """
                                            else:
                                                sql = """
                                                INSERT INTO violations (id, tenant_id, camera_id, violation_timestamp, face_id, violation_type, violation_image_path, details)
                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                """
                                            
                                            # For the is_first_time flag, check if this is the very first time
                                            # (not using local_violation_timers which could be reset)
                                            elapsed_seconds = 0
                                            if key in local_violation_timers:
                                                elapsed_seconds = current_time - local_violation_timers[key]
                                                
                                            # This is more accurate for is_first_time
                                            is_first_time = len([v for v in previously_detected_violations if v == key]) == 0
                                            
                                            violation_details = json.dumps({
                                                "person_id": person_id,
                                                "elapsed_seconds": elapsed_seconds,
                                                "face_detected": embedding is not None,
                                                "is_first_time": is_first_time
                                            })
                                            
                                            cursor.execute(sql, (
                                                violation_id,
                                                tenant_id, 
                                                camera_id,
                                                current_time,
                                                face_id,
                                                vio,
                                                output_path,
                                                violation_details
                                            ))
                                            
                                            conn.commit()
                                            conn.close()
                                            
                                            # Update violation timer and increment count
                                            local_violation_timers[key] = current_time
                                            increment_violations_detected(video_id)
                                            logging.info(f"[SUCCESS] Violation recorded in DB with ID {violation_id}: {vio} for face {face_id}")
                                        
                                            # Trigger external event if configured
                                            if config.get("external_trigger_url"):
                                                try:
                                                    requests.post(config["external_trigger_url"], json={
                                                        "id": violation_id,
                                                        "tenant_id": tenant_id,
                                                        "camera_id": camera_id,
                                                        "violation_timestamp": current_time,
                                                        "face_id": face_id,
                                                        "violation_type": vio,
                                                        "violation_image_path": output_path,
                                                        "details": violation_details
                                                    }, timeout=5)
                                                except Exception as e:
                                                    logging.error(f"[ERROR] Failed to trigger external event: {str(e)}")
                                        except Exception as db_error:
                                            logging.error(f"[ERROR] Failed to save violation to database: {str(db_error)}")
                                            traceback.print_exc()
                        except Exception as face_error:
                            logging.error(f"[ERROR] Failed to process face: {str(face_error)}")
                            continue
                                
                    # Clean up violation timers - only for this frame's face_ids
                    detected_face_ids = set()
                    for person_id, (x1, y1, x2, y2) in violation_faces:
                        try:
                            if embedding is not None:
                                matched_face_ids = find_matching_faces(tenant_id, embedding, similarity_threshold)
                                if matched_face_ids:
                                    detected_face_ids.add(matched_face_ids[0])
                        except:
                            pass
                    
                    # If face was detected but isn't in this frame, remove its timers
                    for key in list(local_violation_timers.keys()):
                        this_tenant, this_camera, this_face, this_vio = key
                        if (this_tenant == tenant_id and this_camera == camera_id and 
                            (this_face not in detected_face_ids or this_vio not in violations_found)):
                            logging.info(f"Removing timer for {key} - not detected in current frame")
                            local_violation_timers.pop(key, None)
                            # Note: we don't remove from first_time_violations since those are persistent records
                
                # Increment frame counter
                increment_frames_processed(video_id, frame_skip_interval)
                
            except Exception as e:
                logging.error(f"[ERROR] Error processing frame {frame_count}: {str(e)}")
                continue

            # Check tenant status
            config = get_tenant_config(tenant_id)
            if not config or not config.get("is_active", False):
                status_queue.put(("stopped", f"Tenant {tenant_id} became inactive"))
                update_video_status(video_id, "stopped")
                break

    except Exception as e:
        logging.error(f"[ERROR] Fatal error in process_video: {str(e)}")
        status_queue.put(("error", str(e)))
        update_video_status(video_id, "error")
    finally:
        if cap:
            cap.release()
        logging.info(f"Finished processing video {video_id}")
        
        # Log violation stats at end of processing
        try:
            log_violation_stats(video_id, tenant_id, camera_id)
        except Exception as e:
            logging.error(f"[ERROR] Error logging violation stats: {str(e)}")
        
        # Only update to done if processing wasn't stopped for other reasons
        video_record = get_video_record(video_id)
        if video_record and video_record.get("status") == "processing":
            update_video_status(video_id, "done")

def log_violation_stats(video_id, tenant_id, camera_id):
    """Log summary statistics of violations for the given video"""
    try:
        conn = get_sqlite_connection()
        c = conn.cursor()
        
        # Count violations by type and employee
        is_postgres = is_using_postgres()
        if is_postgres:
            query = """
            SELECT violation_type, face_id, COUNT(*) 
            FROM violations 
            WHERE tenant_id = %s AND camera_id = %s
            GROUP BY violation_type, face_id
            """
            c.execute(query, (tenant_id, camera_id))
        else:
            query = """
            SELECT violation_type, face_id, COUNT(*) 
            FROM violations 
            WHERE tenant_id = ? AND camera_id = ?
            GROUP BY violation_type, face_id
            """
            c.execute(query, (tenant_id, camera_id))
            
        violations_by_type = {}
        for row in c.fetchall():
            vio_type, face_id, count = row
            if vio_type not in violations_by_type:
                violations_by_type[vio_type] = {}
            violations_by_type[vio_type][face_id] = count
        
        logging.info(f"Violation statistics for video {video_id} (tenant: {tenant_id}, camera: {camera_id}):")
        for vio_type, faces in violations_by_type.items():
            logging.info(f"  {vio_type}: {len(faces)} employees, {sum(faces.values())} total violations")
            for face_id, count in faces.items():
                logging.info(f"    - {face_id}: {count} violations")
        
    except Exception as e:
        logging.error(f"[ERROR] Error logging violation stats: {str(e)}")
    finally:
        conn.close()

def start_video_processing(video_id, tenant_id, camera_id, similarity_threshold, violation_timers, violation_timers_lock):
    status_queue = Queue()
    process = Process(
        target=process_video,
        args=(video_id, tenant_id, camera_id, similarity_threshold, 
              violation_timers, violation_timers_lock, status_queue)
    )
    process.start()
    return process, status_queue 