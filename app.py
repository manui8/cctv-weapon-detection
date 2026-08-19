import streamlit as st
import cv2
import os
import math
import tempfile
import subprocess
import shutil
import json
from datetime import datetime

import numpy as np
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="WeaponGuard AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "best.pt"

CONFIDENCE_THRESHOLD = 0.60
REQUIRED_CONSECUTIVE_FRAMES = 5

TEAM_LEAD = "S Nagasindhu"

TEAM_MEMBERS = [
    "S Bhavyasri",
    "S Manasa",
    "S Anusha"
]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ================================
       GENERAL
       ================================ */

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* ================================
       SIDEBAR
       ================================ */

    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
    }

    .sidebar-heading {
        font-size: 18px;
        font-weight: bold;
    }


    /* ================================
       HOME PAGE
       ================================ */

    .home-title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-top: 20px;
        margin-bottom: 35px;
    }

    .project-main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-top: 20px;
        margin-bottom: 35px;
    }

    .description-box {
        background-color: #191d27;
        border: 1px solid #303644;
        border-radius: 16px;
        padding: 30px 38px;
        margin: 0 auto 30px auto;
        max-width: 950px;
    }

    .description-heading {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin-bottom: 18px;
    }

    .description-text {
        text-align: center;
        font-size: 18px;
        line-height: 1.8;
        color: #dddddd;
    }


    /* ================================
       DETECTION PAGE
       ================================ */

    .detection-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-bottom: 8px;
    }

    .detection-subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .detection-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 15px;
    }

    .safe {
        background-color: #123d24;
        color: #50fa7b;
    }

    .danger {
        background-color: #4a1111;
        color: #ff5555;
    }

    .info-card {
        background-color: #1b1f2a;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
    }

    .video-heading {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# =========================================================
# HELPER - WEAPON CLASS CHECK
# =========================================================

def is_weapon_class(class_name):
    """
    Supports common class names used in weapon datasets.
    """

    name = str(class_name).lower().strip()

    weapon_keywords = [
        "weapon",
        "gun",
        "pistol",
        "rifle",
        "firearm",
        "knife",
        "sword",
        "shotgun",
        "handgun",
        "revolver"
    ]

    for keyword in weapon_keywords:
        if keyword in name:
            return True

    return False


# =========================================================
# HELPER - BOX CENTER
# =========================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


# =========================================================
# HELPER - DISTANCE
# =========================================================

def distance(box1, box2):

    c1 = box_center(box1)
    c2 = box_center(box2)

    return math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )


# =========================================================
# SAVE DETECTION DATA
# =========================================================

def save_detection_data(
    detection_type,
    weapon_count,
    total_objects=0
):

    data_file = "detection_log.json"

    detection_record = {
        "timestamp": datetime.now().isoformat(),
        "type": detection_type,
        "weapons_detected": weapon_count,
        "total_objects": total_objects
    }

    try:

        if os.path.exists(data_file):

            with open(data_file, "r") as f:
                data = json.load(f)

        else:

            data = {
                "detections": []
            }

        data["detections"].append(
            detection_record
        )

        if len(data["detections"]) > 100:

            data["detections"] = (
                data["detections"][-100:]
            )

        with open(data_file, "w") as f:

            json.dump(
                data,
                f,
                indent=2
            )

    except Exception:
        pass


# =========================================================
# ALERT SOUND
# =========================================================

def show_alert_sound():

    alert_path = "alert.mp3"

    if os.path.exists(alert_path):

        with open(alert_path, "rb") as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/mp3",
                autoplay=False
            )


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-title">
                Artificial Intelligence<br>
                Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            "### 🎓 College"
        )

        st.write(
            "VSM College of Engineering"
        )

        st.markdown("---")

        st.markdown(
            "### 👨‍🏫 Project Guide"
        )

        st.write(
            "Mr. Abdul Aziz MD"
        )

        st.markdown("---")

        st.markdown(
            "### 👥 Team Members"
        )

        st.write(
            f"⭐ **{TEAM_LEAD} — Team Lead**"
        )

        for member in TEAM_MEMBERS:

            st.write(
                f"• {member} — Team Member"
            )

    # =====================================================
    # MAIN TITLE
    # =====================================================

    st.markdown(
        """
        <div class="project-main-title">
            🚨 WeaponGuard AI
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # DESCRIPTION
    # =====================================================

    st.markdown(
        """
        <div class="description-box">

            <div class="description-heading">
                Description
            </div>

            <div class="description-text">
                WeaponGuard AI is an intelligent CCTV security
                system designed to automatically detect weapons
                from images and video footage. Using Artificial
                Intelligence and YOLO-based object detection,
                the system analyzes visual data, identifies
                suspicious weapons, highlights detected objects
                with bounding boxes, and provides an alert when
                a weapon is found. The system helps improve
                surveillance, supports faster security response,
                and provides an efficient approach for detecting
                potential threats in CCTV environments.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # NEXT BUTTON
    # =====================================================

    if st.button(
        "➡️ NEXT",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.page = "detection"

        st.rerun()


# =========================================================
# IMAGE DETECTION
# =========================================================

def image_detection(
    model,
    confidence_threshold
):

    st.subheader(
        "📷 Image Detection"
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ],
        key="image_upload"
    )

    if uploaded_image is None:

        return

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    image_array = np.array(
        image
    )

    st.success(
        "✅ Image uploaded successfully"
    )

    if st.button(
        "🔍 Detect Weapon",
        use_container_width=True,
        key="image_detect_button"
    ):

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Run model with low threshold first
        results = model(
            image_bgr,
            conf=0.10,
            verbose=False
        )

        output_image = image_bgr.copy()

        weapon_count = 0
        total_objects = 0

        # =================================================
        # DETECTION
        # =================================================

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                total_objects += 1

                # -----------------------------------------
                # WEAPON CHECK
                # -----------------------------------------

                if (
                    is_weapon_class(class_name)
                    and confidence >= confidence_threshold
                ):

                    weapon_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Bounding box
                    cv2.rectangle(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    # Label
                    label = (
                        f"WEAPON "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        output_image,
                        label,
                        (
                            x1,
                            max(y1 - 10, 30)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

        # =================================================
        # DISPLAY
        # =================================================

        output_rgb = cv2.cvtColor(
            output_image,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            st.markdown(
                '<div class="video-heading">📥 Input Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_array,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="video-heading">📤 Detected Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                output_rgb,
                use_container_width=True
            )

        # =================================================
        # SAVE LOG
        # =================================================

        save_detection_data(
            "image",
            weapon_count,
            total_objects
        )

        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        if weapon_count > 0:

            st.markdown(
                f"""
                <div class="detection-box danger">
                    🚨 {weapon_count} WEAPON(S) DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

            # Image alert
            show_alert_sound()

        else:

            st.markdown(
                """
                <div class="detection-box safe">
                    ✅ NO WEAPON DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # DOWNLOAD
        # =================================================

        output_pil = Image.fromarray(
            output_rgb
        )

        output_bytes = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        )

        output_pil.save(
            output_bytes.name
        )

        output_bytes.close()

        with open(
            output_bytes.name,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Detected Image",
                data=file.read(),
                file_name="weapon_detection_output.png",
                mime="image/png",
                use_container_width=True
            )


# =========================================================
# CONVERT VIDEO TO PLAYABLE MP4
# =========================================================

def convert_to_playable_mp4(
    input_path,
    output_path
):

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:

        return False

    try:

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-acodec",
            "aac",
            "-movflags",
            "+faststart",
            output_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return (
            result.returncode == 0
            and os.path.exists(output_path)
        )

    except Exception:

        return False


# =========================================================
# VIDEO DETECTION
# =========================================================

def video_detection(
    model,
    confidence_threshold,
    required_frames
):

    st.subheader(
        "🎥 CCTV Video Detection"
    )

    uploaded_video = st.file_uploader(
        "Upload CCTV video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_upload"
    )

    if uploaded_video is None:

        return

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.read()
    )

    input_file.close()

    input_video_path = input_file.name

    # =====================================================
    # SHOW INPUT VIDEO
    # =====================================================

    st.markdown(
        '<div class="video-heading">📥 Input CCTV Video</div>',
        unsafe_allow_html=True
    )

    # Read uploaded bytes again
    uploaded_video.seek(0)

    original_video_bytes = uploaded_video.read()

    st.video(
        original_video_bytes
    )

    # =====================================================
    # START BUTTON
    # =====================================================

    start_detection = st.button(
        "🔍 Start Weapon Detection",
        use_container_width=True,
        type="primary",
        key="video_detect_button"
    )

    if not start_detection:

        return

    # =====================================================
    # VIDEO CAPTURE
    # =====================================================

    cap = cv2.VideoCapture(
        input_video_path
    )

    if not cap.isOpened():

        st.error(
            "❌ Could not open uploaded video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # =====================================================
    # TEMP OUTPUT
    # =====================================================

    raw_output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )

    raw_output_path = raw_output_file.name

    raw_output_file.close()

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )

    out = cv2.VideoWriter(
        raw_output_path,
        fourcc,
        fps,
        (width, height)
    )

    # =====================================================
    # TRACKING VARIABLES
    # =====================================================

    frame_number = 0

    consecutive_weapon_frames = {}

    last_boxes = {}

    confirmed_detections = 0

    detection_events = []

    weapon_detected_in_video = False

    # =====================================================
    # UI
    # =====================================================

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    preview_placeholder = st.empty()

    # =====================================================
    # PROCESS VIDEO
    # =====================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        current_weapons = []

        # =================================================
        # YOLO
        # =================================================

        results = model(
            frame,
            conf=0.10,
            verbose=False
        )

        # =================================================
        # FIND WEAPONS
        # =================================================

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                if (
                    is_weapon_class(class_name)
                    and confidence >= confidence_threshold
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    current_weapons.append(
                        {
                            "box": (
                                x1,
                                y1,
                                x2,
                                y2
                            ),
                            "confidence": confidence
                        }
                    )

        # =================================================
        # TRACK WEAPONS
        # =================================================

        confirmed_weapon_boxes = []

        new_last_boxes = {}

        used_previous_ids = set()

        for weapon in current_weapons:

            weapon_box = weapon["box"]
            weapon_conf = weapon["confidence"]

            best_match_id = None
            best_match_distance = 150

            for (
                previous_id,
                previous_box
            ) in last_boxes.items():

                if previous_id in used_previous_ids:
                    continue

                movement = distance(
                    previous_box,
                    weapon_box
                )

                if movement < best_match_distance:

                    best_match_distance = movement
                    best_match_id = previous_id

            # Existing weapon
            if best_match_id is not None:

                used_previous_ids.add(
                    best_match_id
                )

                consecutive_weapon_frames[
                    best_match_id
                ] = (
                    consecutive_weapon_frames.get(
                        best_match_id,
                        0
                    ) + 1
                )

                new_last_boxes[
                    best_match_id
                ] = weapon_box

                if (
                    consecutive_weapon_frames[
                        best_match_id
                    ] >= required_frames
                ):

                    confirmed_weapon_boxes.append(
                        {
                            "box": weapon_box,
                            "confidence": weapon_conf,
                            "id": best_match_id
                        }
                    )

            # New weapon
            else:

                new_id = (
                    max(
                        consecutive_weapon_frames.keys(),
                        default=-1
                    ) + 1
                )

                consecutive_weapon_frames[
                    new_id
                ] = 1

                new_last_boxes[
                    new_id
                ] = weapon_box

        last_boxes = new_last_boxes

        # =================================================
        # DRAW DETECTIONS
        # =================================================

        if confirmed_weapon_boxes:

            weapon_detected_in_video = True

            confirmed_detections += len(
                confirmed_weapon_boxes
            )

            for detection in confirmed_weapon_boxes:

                x1, y1, x2, y2 = detection["box"]

                confidence = detection[
                    "confidence"
                ]

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                # Label
                label = (
                    f"WEAPON "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                time_sec = (
                    frame_number / fps
                )

                detection_events.append(
                    {
                        "time": round(
                            time_sec,
                            2
                        ),
                        "confidence": round(
                            confidence,
                            2
                        )
                    }
                )

            # Big warning
            cv2.rectangle(
                frame,
                (10, 10),
                (width - 10, 75),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "!!! WEAPON DETECTED !!!",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                3
            )

        else:

            cv2.putText(
                frame,
                "No Weapon Detected",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # =================================================
        # WRITE FRAME
        # =================================================

        out.write(
            frame
        )

        # =================================================
        # LIVE PREVIEW
        # =================================================

        preview_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        preview_placeholder.image(
            preview_rgb,
            caption="🔄 Processing CCTV Video...",
            width=700
        )

        # =================================================
        # PROGRESS
        # =================================================

        if total_frames > 0:

            progress = (
                frame_number /
                total_frames
            )

            progress_bar.progress(
                min(progress, 1.0)
            )

        status_text.write(
            f"Processing frame "
            f"{frame_number} / "
            f"{total_frames}"
        )

    # =====================================================
    # RELEASE
    # =====================================================

    cap.release()

    out.release()

    progress_bar.progress(
        1.0
    )

    status_text.success(
        "✅ Video processing completed!"
    )

    # Remove live preview
    preview_placeholder.empty()

    # =====================================================
    # CONVERT TO PLAYABLE MP4
    # =====================================================

    playable_video_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    playable_video_path = (
        playable_video_file.name
    )

    playable_video_file.close()

    converted = convert_to_playable_mp4(
        raw_output_path,
        playable_video_path
    )

    # =====================================================
    # IF FFMPEG IS NOT AVAILABLE
    # =====================================================

    if not converted:

        playable_video_path = raw_output_path

    # =====================================================
    # READ OUTPUT VIDEO
    # =====================================================

    if os.path.exists(
        playable_video_path
    ):

        with open(
            playable_video_path,
            "rb"
        ) as video_file:

            output_video_bytes = (
                video_file.read()
            )

    else:

        output_video_bytes = None

    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📊 Detection Results"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>🎞️ Frames</h3>
                <h2>{total_frames}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="info-card">
                <h3>🚨 Detections</h3>
                <h2>{confirmed_detections}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        if weapon_detected_in_video:

            st.markdown(
                """
                <div class="detection-box danger">
                    🚨 WEAPON DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="detection-box safe">
                    ✅ NO WEAPON DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

    # =====================================================
    # SIREN ONLY IF VIDEO HAS WEAPON
    # =====================================================

    if weapon_detected_in_video:

        st.markdown("---")

        st.warning(
            "🚨 Weapon detected in the CCTV video. "
            "Alert sound is enabled below."
        )

        show_alert_sound()

    # =====================================================
    # INPUT + OUTPUT VIDEO SIDE BY SIDE
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🎥 CCTV Video Comparison"
    )

    video_col1, video_col2 = st.columns(
        2,
        gap="large"
    )

    with video_col1:

        st.markdown(
            '<div class="video-heading">📥 Original Video</div>',
            unsafe_allow_html=True
        )

        st.video(
            original_video_bytes
        )

    with video_col2:

        st.markdown(
            '<div class="video-heading">📤 Detected Output Video</div>',
            unsafe_allow_html=True
        )

        if output_video_bytes is not None:

            st.video(
                output_video_bytes
            )

        else:

            st.error(
                "❌ Output video could not be generated."
            )

    # =====================================================
    # DOWNLOAD OUTPUT
    # =====================================================

    if output_video_bytes is not None:

        st.download_button(
            label="⬇️ Download Processed Video",
            data=output_video_bytes,
            file_name="weapon_detection_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # =====================================================
    # DETECTION EVENTS
    # =====================================================

    if detection_events:

        st.markdown("---")

        st.subheader(
            "🚨 Detection Events"
        )

        # Show unique events approximately
        displayed_times = set()

        for event in detection_events:

            event_time = event["time"]

            rounded_time = round(
                event_time,
                1
            )

            if rounded_time in displayed_times:
                continue

            displayed_times.add(
                rounded_time
            )

            st.warning(
                f"⏱️ Time: {event_time} sec | "
                f"Confidence: {event['confidence']}"
            )

            if len(displayed_times) >= 20:
                break

    # =====================================================
    # SAVE LOG
    # =====================================================

    save_detection_data(
        "video",
        confirmed_detections,
        total_frames
    )

    # =====================================================
    # CLEAN TEMP FILES
    # =====================================================

    try:

        if os.path.exists(
            input_video_path
        ):

            os.remove(
                input_video_path
            )

    except Exception:
        pass


# =========================================================
# DETECTION PAGE
# =========================================================

def detection_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-title">
                ⚙️ DETECTION SETTINGS
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # =================================================
        # BACK BUTTON
        # =================================================

        if st.button(
            "⬅️ BACK",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=CONFIDENCE_THRESHOLD,
            step=0.05
        )

        required_frames = st.slider(
            "🎞️ Required Consecutive Frames",
            min_value=1,
            max_value=15,
            value=REQUIRED_CONSECUTIVE_FRAMES,
            step=1
        )

        st.markdown("---")

        st.write(
            "🤖 **Model:** YOLO"
        )

        st.write(
            "🎯 **Detection:** Weapon"
        )

        st.write(
            "📹 **Input:** Image / CCTV Video"
        )

        st.markdown("---")

        st.markdown(
            "### 👥 Team"
        )

        st.write(
            f"⭐ **{TEAM_LEAD} — Team Lead**"
        )

        for member in TEAM_MEMBERS:

            st.write(
                f"• {member} — Team Member"
            )

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        st.error(
            "❌ best.pt not found!"
        )

        st.info(
            "Please place best.pt in the same "
            "folder as app.py."
        )

        st.stop()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = load_model()

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        """
        <div class="detection-title">
            🚨 WeaponGuard AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="detection-subtitle">
            AI-powered weapon detection using YOLO
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection"
        ]
    )

    # =====================================================
    # IMAGE TAB
    # =====================================================

    with tab1:

        image_detection(
            model,
            confidence_threshold
        )

    # =====================================================
    # VIDEO TAB
    # =====================================================

    with tab2:

        video_detection(
            model,
            confidence_threshold,
            required_frames
        )


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    home_page()

else:

    detection_page()
