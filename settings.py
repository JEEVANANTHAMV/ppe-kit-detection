from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]
ASSET_DIR = ROOT / "images"
LOGO_PATH = ASSET_DIR / "logo.png"
# Images config
IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# Videos config
VIDEO_DIR = ROOT / "videos"
VIDEOS_DICT = {
    "video_1": VIDEO_DIR / "video_1.mp4",
    "video_2": VIDEO_DIR / "video_2.mp4",
    "video_3": VIDEO_DIR / "video_3.mp4",
}
DEFAULT_VIDEO = VIDEO_DIR / "video_1.mp4"
DEFAULT_DETECT_VIDEO = VIDEO_DIR / "video_1.mp4"
DEFAULT_YOUTUBE = "https://www.youtube.com/watch?v=UZ_RrLxMrEg"
# ML Model config
MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "weights.onnx"

SEGMENTATION_MODEL = MODEL_DIR / "yolov8n-seg.pt"

# Webcam
WEBCAM_PATH = 0
