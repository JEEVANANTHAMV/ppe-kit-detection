import pytest
import os
import json
from fastapi.testclient import TestClient
from app import app
import cv2
import numpy as np
from helper import get_connection, format_query
import shutil
import tempfile
import helper

# Set testing mode
os.environ["TESTING"] = "true"

client = TestClient(app)

# Test data
TEST_TENANT_ID = "test_tenant_1"
TEST_CAMERA_ID = "test_camera_1"
TEST_FACE_ID = "test_face_1"
TEST_TENANT_NAME = "Test Tenant"

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    # Setup
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("violation_images", exist_ok=True)
    
    # Create test video file
    test_video_path = "uploads/test_video.mp4"
    if not os.path.exists(test_video_path):
        # Create a dummy video file
        cap = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for _ in range(30):  # 1 second of video
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cap.write(frame)
        cap.release()
    
    # Create test face image
    test_face_path = "uploads/test_face.jpg"
    if not os.path.exists(test_face_path):
        # Create a dummy face image
        face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(test_face_path, face_img)
    
    yield
    
    # Cleanup
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
    if os.path.exists(test_face_path):
        os.remove(test_face_path)
    shutil.rmtree("violation_images", ignore_errors=True)
    
    # Clean up database
    conn, db_type = get_connection()
    c = conn.cursor()
    tables = ["violations", "faces", "videos", "tenant_configs"]
    for table in tables:
        c.execute(format_query(f"DELETE FROM {table}", db_type))
    conn.commit()
    conn.close()

@pytest.fixture(autouse=True)
def setup_database():
    """Set up the database before each test."""
    helper.ensure_tables_exist()
    # Add a test tenant config
    tenant_config = {
        "name": "Test Tenant",
        "similarity_threshold": 0.7,
        "ppe_thresholds": {
            "no_mask": 70,
            "no_safety_vest": 70,
            "no_hardhat": 70
        }
    }
    helper.add_tenant_config("test-tenant", json.dumps(tenant_config))
    yield
    # Clean up database after test
    if os.path.exists("test.db"):
        pass # Don't delete as it could affect other tests

def setup_module():
    """Set up test environment."""
    # Create a temp directory for uploads
    os.makedirs("uploads", exist_ok=True)
    # Ensure database tables exist
    helper.ensure_tables_exist()
    # Add a test tenant config
    tenant_config = {
        "name": "Test Tenant",
        "similarity_threshold": 0.7,
        "ppe_thresholds": {
            "no_mask": 70,
            "no_safety_vest": 70,
            "no_hardhat": 70
        }
    }
    helper.add_tenant_config("test-tenant", json.dumps(tenant_config))

def test_tenant_config_endpoints():
    # Add tenant config
    tenant_config = {
        "name": "Test Tenant 2",
        "similarity_threshold": 0.7,
        "ppe_thresholds": {
            "no_mask": 70,
            "no_safety_vest": 70,
            "no_hardhat": 70
        }
    }
    response = client.post(
        "/tenants",
        data={"tenant_id": "test-tenant-2", "config": json.dumps(tenant_config)}
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "test-tenant-2"
    
    # Get tenant config
    response = client.get("/tenants/test-tenant-2")
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "test-tenant-2"
    assert json.loads(response.json()["config"])["name"] == "Test Tenant 2"
    
    # List tenant configs
    response = client.get("/tenants")
    assert response.status_code == 200
    assert len(response.json()["tenants"]) >= 2  # At least the two we added
    
    # Update tenant config
    updated_config = {
        "name": "Updated Test Tenant",
        "similarity_threshold": 0.8,
        "ppe_thresholds": {
            "no_mask": 80,
            "no_safety_vest": 80,
            "no_hardhat": 80
        }
    }
    response = client.put(
        "/tenants/test-tenant-2",
        data={"config": json.dumps(updated_config)}
    )
    assert response.status_code == 200
    
    # Verify update
    response = client.get("/tenants/test-tenant-2")
    assert response.status_code == 200
    assert json.loads(response.json()["config"])["name"] == "Updated Test Tenant"
    
    # Delete tenant config
    response = client.delete("/tenants/test-tenant-2")
    assert response.status_code == 200
    
    # Verify deletion
    response = client.get("/tenants/test-tenant-2")
    assert response.status_code == 404

def test_video_endpoints():
    # Test with a temporary video file
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(b"fake video content")
        temp_video.flush()
        temp_video.seek(0)
        
        # Upload video
        response = client.post(
            "/videos",
            data={"tenant_id": "test-tenant", "camera_id": "test-camera"},
            files={"file": ("test_video.mp4", temp_video, "video/mp4")}
        )
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "test-tenant"
        assert response.json()["camera_id"] == "test-camera"
        assert response.json()["status"] == "pending"
        
        # List videos
        response = client.get("/videos?tenant_id=test-tenant")
        assert response.status_code == 200
        assert len(response.json()["videos"]) >= 1
        
        # Get specific video
        response = client.get("/videos/test-camera?tenant_id=test-tenant")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "test-tenant"
        assert response.json()["camera_id"] == "test-camera"
        
        # Update video with stream URL
        response = client.put(
            "/videos/test-camera",
            data={
                "tenant_id": "test-tenant",
                "stream_url": "rtsp://example.com/stream"
            }
        )
        assert response.status_code == 200
        assert response.json()["stream_url"] == "rtsp://example.com/stream"
        
        # Update video with S3 URL
        response = client.put(
            "/videos/test-camera",
            data={
                "tenant_id": "test-tenant",
                "s3_url": "s3://bucket/video.mp4"
            }
        )
        assert response.status_code == 200
        assert response.json()["s3_url"] == "s3://bucket/video.mp4"
        
        # Get stats
        response = client.get("/stats?tenant_id=test-tenant")
        assert response.status_code == 200
        assert response.json()["total_videos"] >= 1
        
        # Get status
        response = client.get("/status?tenant_id=test-tenant")
        assert response.status_code == 200
        assert len(response.json()["status"]) >= 1
        
        # Delete video
        response = client.delete("/videos/test-camera?tenant_id=test-tenant")
        assert response.status_code == 200
        
        # Verify deletion
        response = client.get("/videos/test-camera?tenant_id=test-tenant")
        assert response.status_code == 404

def test_face_endpoints():
    # Create a fake image file
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
        temp_image.write(b"fake image content")
        temp_image.flush()
        temp_image.seek(0)
        
        # Add face
        response = client.post(
            "/faces",
            data={"tenant_id": "test-tenant", "name": "Test Person"},
            files={"file": ("test_face.jpg", temp_image, "image/jpeg")}
        )
        assert response.status_code == 200
        face_id = response.json()["face_id"]
        assert response.json()["tenant_id"] == "test-tenant"
        assert response.json()["name"] == "Test Person"
        
        # List faces
        response = client.get("/faces?tenant_id=test-tenant")
        assert response.status_code == 200
        assert len(response.json()["faces"]) >= 1
        
        # Get specific face
        response = client.get(f"/faces/{face_id}?tenant_id=test-tenant")
        assert response.status_code == 200
        assert response.json()["face_id"] == face_id
        assert response.json()["tenant_id"] == "test-tenant"
        assert response.json()["name"] == "Test Person"
        
        # Update face with new image
        temp_image.seek(0)
        response = client.put(
            f"/faces/{face_id}",
            data={"tenant_id": "test-tenant", "name": "Updated Person"},
            files={"file": ("updated_face.jpg", temp_image, "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Person"
        
        # Update face with S3 URL
        response = client.put(
            f"/faces/{face_id}",
            data={
                "tenant_id": "test-tenant",
                "name": "S3 Person",
                "s3_url": "s3://bucket/face.jpg"
            }
        )
        assert response.status_code == 200
        assert response.json()["name"] == "S3 Person"
        
        # Delete face
        response = client.delete(f"/faces/{face_id}?tenant_id=test-tenant")
        assert response.status_code == 200
        
        # Verify deletion
        response = client.get(f"/faces/{face_id}?tenant_id=test-tenant")
        assert response.status_code == 404

def test_error_cases():
    # Invalid tenant configuration (missing required fields)
    response = client.post(
        "/tenants",
        data={"tenant_id": "invalid-tenant", "config": json.dumps({})}
    )
    assert response.status_code == 400
    
    # Non-existent tenant
    response = client.get("/tenants/non-existent-tenant")
    assert response.status_code == 404
    
    # Duplicate tenant (add same tenant twice)
    tenant_config = {
        "name": "Duplicate Tenant",
        "similarity_threshold": 0.7,
        "ppe_thresholds": {
            "no_mask": 70,
            "no_safety_vest": 70,
            "no_hardhat": 70
        }
    }
    response = client.post(
        "/tenants",
        data={"tenant_id": "duplicate-tenant", "config": json.dumps(tenant_config)}
    )
    assert response.status_code == 200
    
    response = client.post(
        "/tenants",
        data={"tenant_id": "duplicate-tenant", "config": json.dumps(tenant_config)}
    )
    assert response.status_code == 409
    
    # Upload video without required fields
    response = client.post("/videos", data={})
    assert response.status_code == 422
    
    # Upload face without required fields
    response = client.post("/faces", data={})
    assert response.status_code == 422
    
    # Update non-existent video
    response = client.put(
        "/videos/non-existent-camera",
        data={
            "tenant_id": "test-tenant",
            "stream_url": "rtsp://example.com/stream"
        }
    )
    assert response.status_code == 404
    
    # Update non-existent face
    response = client.put(
        "/faces/non-existent-face",
        data={"tenant_id": "test-tenant", "name": "Non-existent Person"}
    )
    assert response.status_code == 404
    
    # Cleanup
    client.delete("/tenants/duplicate-tenant")

def teardown_module():
    """Clean up after tests."""
    # Remove test database
    if os.path.exists("test.db"):
        os.remove("test.db")
    # Remove uploads directory
    if os.path.exists("uploads"):
        for file in os.listdir("uploads"):
            os.remove(os.path.join("uploads", file))
        os.rmdir("uploads")

def test_status_and_stats_endpoints():
    # Test status endpoint
    response = client.get("/status")
    assert response.status_code == 200
    assert "status" in response.json()
    
    # Test stats endpoint
    response = client.get("/stats")
    assert response.status_code == 200
    assert "total_videos" in response.json()

def test_live_url_endpoint():
    # Test creating a stream URL video
    response = client.post(
        "/videos",
        data={
            "tenant_id": "test-tenant",
            "camera_id": "test-stream-camera",
            "stream_url": "rtsp://example.com/stream"
        }
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "test-tenant"
    assert response.json()["camera_id"] == "test-stream-camera"
    assert response.json()["stream_url"] == "rtsp://example.com/stream"
    
    # Clean up
    client.delete("/videos/test-stream-camera?tenant_id=test-tenant")

def test_process_endpoint():
    # Create test data - add a tenant config, a face, and a video
    # Add a tenant config
    tenant_config = {
        "name": "Process Test Tenant",
        "similarity_threshold": 0.7,
        "ppe_thresholds": {
            "no_mask": 70,
            "no_safety_vest": 70,
            "no_hardhat": 70
        }
    }
    response = client.post(
        "/tenants",
        data={"tenant_id": "process-tenant", "config": json.dumps(tenant_config)}
    )
    assert response.status_code == 200
    
    # Add a face for the tenant
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
        temp_image.write(b"fake image content")
        temp_image.flush()
        temp_image.seek(0)
        
        response = client.post(
            "/faces",
            data={"tenant_id": "process-tenant", "name": "Process Test Person"},
            files={"file": ("process_test_face.jpg", temp_image, "image/jpeg")}
        )
        assert response.status_code == 200
    
    # Add a video for processing
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(b"fake video content")
        temp_video.flush()
        temp_video.seek(0)
        
        response = client.post(
            "/videos",
            data={"tenant_id": "process-tenant", "camera_id": "process-camera"},
            files={"file": ("process_test_video.mp4", temp_video, "video/mp4")}
        )
        assert response.status_code == 200
    
    # Trigger processing
    response = client.post("/process")
    assert response.status_code == 200
    assert "triggered" in response.json() or "message" in response.json()
    
    # Clean up
    client.delete("/videos/process-camera?tenant_id=process-tenant")
    
    # Get the face ID from listing faces
    response = client.get("/faces?tenant_id=process-tenant")
    assert response.status_code == 200
    faces = response.json()["faces"]
    
    if faces:
        face_id = faces[0]["face_id"]
        client.delete(f"/faces/{face_id}?tenant_id=process-tenant")
    
    client.delete("/tenants/process-tenant")

def test_trigger_event_endpoint():
    # Test triggering event
    response = client.post(
        "/trigger-event",
        json={
            "tenant_id": TEST_TENANT_ID,
            "camera_id": TEST_CAMERA_ID,
            "event_url": "http://test.com/webhook"
        }
    )
    assert response.status_code == 200
    assert "status_code" in response.json()
    assert "response_text" in response.json()
    
    # Test invalid event trigger
    response = client.post(
        "/trigger-event",
        json={
            "tenant_id": TEST_TENANT_ID,
            "camera_id": TEST_CAMERA_ID
        }
    )
    assert response.status_code == 400 