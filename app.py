import os
import cv2
import time
import numpy as np
import PIL
import streamlit as st
import tempfile
from inference_sdk import InferenceHTTPClient
import settings
import helper
from pytubefix import YouTube  # Ensure this package is installed

# Setting page layout
st.set_page_config(
    page_title="PPE Kit Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("PPE Kit Detection")

# Initialize inference client
CLIENT = InferenceHTTPClient(api_url="http://localhost:9001")

# Sidebar settings
st.sidebar.header("ML Model Config")
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

st.sidebar.header("Image/Video/YouTube")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

# -------------------------
# IMAGE PROCESSING BRANCH
# -------------------------
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    col1, col2 = st.columns(2)

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image, caption="Default Image", use_container_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image, caption="Detected Image", use_container_width=True)
        else:
            if st.sidebar.button("Detect Objects"):
                # Convert PIL to OpenCV (BGR)
                uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
                res = CLIENT.infer(uploaded_image_cv, model_id="construction-site-safety/27")
                res_plotted = helper.map_layout(image=uploaded_image_cv, result=res)
                # Convert to RGB for display
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_plotted_rgb, caption="Detected Image", use_container_width=True)

# -------------------------
# VIDEO PROCESSING BRANCH (Local Video)
# -------------------------
elif source_radio == settings.VIDEO:
    source_video = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov", "wmv"))
    col1, col2 = st.columns(2)
    
    with col1:
        if source_video is None:
            default_video_path = str(settings.DEFAULT_VIDEO)
            st.video(default_video_path, format="video/mp4")
        else:
            st.video(source_video, format="video/mp4")
    
    with col2:
        if source_video is None:
            default_detected_video_path = str(settings.DEFAULT_DETECT_VIDEO)
            st.video(default_detected_video_path, format="video/mp4")
        else:
            if st.sidebar.button("Detect Objects"):
                # Save the uploaded video to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(source_video.read())
                temp_file.close()
                video_path = temp_file.name

                st.sidebar.info("Processing video stream...")
                cap = cv2.VideoCapture(video_path)
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                target_fps = 5
                skip_factor = int(original_fps / target_fps) if original_fps > target_fps else 1
                # st.sidebar.text(f"Original FPS: {original_fps:.2f} | Processing every {skip_factor} frame(s)")
                
                st_frame = st.empty()
                frame_count = 0
                processed_frame = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        break
                    # Process only every skip_factor-th frame
                    if frame_count % skip_factor == 0:
                        res = CLIENT.infer(frame, model_id="construction-site-safety/27")
                        annotated_frame = helper.map_layout(image=frame, result=res)
                        # Convert BGR to RGB for display
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                        processed_frame += 1
                        # st.sidebar.text(f"Processed frame: {processed_frame}")
                    frame_count += 1
                    # Maintain approximate playback rate of target_fps
                    time.sleep(1 / target_fps)

# -------------------------
# YOUTUBE VIDEO PROCESSING BRANCH
# -------------------------
elif source_radio == settings.YOUTUBE:
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if youtube_url:
        if not (youtube_url.startswith("https://www.youtube.com") or youtube_url.startswith("https://youtu.be")):
            st.error("Please enter a valid YouTube URL.")
        else:
            progress_bar = st.sidebar.progress(0)

            def progress_function(stream, chunk, bytes_remaining):
                total_size = stream.filesize
                bytes_downloaded = total_size - bytes_remaining
                progress = int((bytes_downloaded / total_size) * 100)
                progress_bar.progress(progress)

            yt = YouTube(youtube_url, on_progress_callback=progress_function)
            stream_obj = yt.streams.filter(file_extension="mp4").first()
            temp_dir = tempfile.gettempdir()
            video_filename = "downloaded_video.mp4"
            video_path = os.path.join(temp_dir, video_filename)
            stream_obj.download(output_path=temp_dir, filename=video_filename)
            st.sidebar.success("Download complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.video(video_path, format="video/mp4")
            with col2:
                if st.sidebar.button("Detect Objects"):
                    st.sidebar.info("Processing YouTube video stream...")
                    cap = cv2.VideoCapture(video_path)
                    original_fps = cap.get(cv2.CAP_PROP_FPS)
                    target_fps = 1
                    skip_factor = int(original_fps / target_fps) if original_fps > target_fps else 1
                    # st.sidebar.text(f"Original FPS: {original_fps:.2f} | Processing every {skip_factor} frame(s)")
                    
                    st_frame = st.empty()
                    frame_count = 0
                    processed_frame = 0

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            cap.release()
                            break
                        if frame_count % skip_factor == 0:
                            res = CLIENT.infer(frame, model_id="construction-site-safety/27")
                            annotated_frame = helper.map_layout(image=frame, result=res)
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                            processed_frame += 1
                            # st.sidebar.text(f"Processed frame: {processed_frame}")
                        frame_count += 1
                        time.sleep(1 / target_fps)
