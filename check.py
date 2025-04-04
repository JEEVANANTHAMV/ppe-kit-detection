import requests
import json
import time

def test_trigger():
    url = "http://localhost:8000/trigger"
    
    payload = {
        "id": "test-123",
        "tenant_id": "tenant-1",
        "camera_id": "cam-1",
        "violation_timestamp": time.time(),
        "face_id": "face-123",
        "violation_type": "NO_HELMET",
        "violation_image_path": "/path/to/image.jpg",
        "details": json.dumps({"additional": "info"})
    }

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_trigger()