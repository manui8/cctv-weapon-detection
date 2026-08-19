import streamlit as st
import cv2
import os
import math
import tempfile
import subprocess
import shutil
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
from datetime import datetime


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

TEAM_LEAD = "S. Nagasindhu"

TEAM_MEMBERS = [
    "S. Bhavyasri",
    "S. Manasa",
    "S. Anusha"
]

GUIDE_NAME = "Mr. Abdul Aziz MD"
COLLEGE_NAME = "VSM College of Engineering"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GENERAL
       ===================================================== */

    .stApp {
        background-color: #0e1117;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #151923;
    }

    .sidebar-aicw {
        font-size: 25px;
        font-weight: 800;
        color: white;
        text-align: center;
        line-height: 1.35;
        padding: 20px 8px 25px 8px;
    }

    .sidebar-line {
        height: 1px;
        background-color: #3a3f4b;
        margin: 8px 0 25px 0;
    }

    .sidebar-heading {
        font-size: 18px;
        font-weight: 700;
        color: #4da6ff;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    .sidebar-text {
        font-size: 16px;
        color: #eeeeee;
        line-height: 1.7;
    }

    .sidebar-team-lead {
        color: #50fa7b;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 5px;
    }


    /* =====================================================
       WELCOME PAGE
       ===================================================== */

    .welcome-title {
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 35px;
    }

    .description-card {
        background-color: #191e29;
        border: 1px solid #2d3442;
        border-radius: 16px;
        padding: 35px 45px;
        margin: 0 auto;
        max-width: 950px;
    }

    .description-heading {
        text-align: center;
        color: #4da6ff;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .description-text {
        color: #eeeeee;
        font-size: 19px;
        line-height: 1.8;
        text-align: justify;
    }

    .next-space {
        height: 30px;
    }


    /* =====================================================
       DETECTION PAGE
       ===================================================== */

    .detection-title {
        text-align: center;
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
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
        margin-bottom: 15px;
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
        background-color: #191e29;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2d3442;
    }

    .video-title {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
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
# SIDEBAR - ONLY ONE TIME
# =========================================================

def show_home_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-aicw">
                Artificial Intelligence<br>
                Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-line"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-heading">🎓 College</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="sidebar-text">{COLLEGE_NAME}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-line"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-heading">👥 Team Members</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sidebar-team-lead">
                ⭐ {TEAM_LEAD} — Team Lead
            </div>

            <div class="sidebar-text">
                • {TEAM_MEMBERS[0]} — Team Member<br>
                • {TEAM_MEMBERS[1]} — Team Member<br>
                • {TEAM_MEMBERS[2]} — Team Member
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-line"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-heading">👨‍🏫 Project Guide</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="sidebar-text">{GUIDE_NAME}</div>',
            unsafe_allow_html=True
        )


# =========================================================
# SIDEBAR - DETECTION PAGE
# =========================================================

def show_detection_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-aicw">
                Artificial Intelligence<br>
                Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-line"></div>',
            unsafe_allow_html=True
        )

        if st.button(
            "⬅️ Back",
            use_container_width=True
        ):

            st.session_state.page = "home"
            st.rerun()

        st.markdown(
            '<div class="sidebar-line"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-heading">⚙️ Detection Settings</div>',
            unsafe_allow_html=True
        )

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

        st.markdown(
            '<div class="sidebar-line"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="sidebar-text">
                🤖 <b>Model:</b> YOLO<br>
                🎯 <b>Detection:</b> Weapon<br>
                📷 <b>Input:</b> Image / CCTV Video
            </div>
            """,
            unsafe_allow_html=True
        )

    return confidence_threshold, required_frames


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

            data["detections"] = \
                data["detections"][-100:]

        with open(data_file, "w") as f:

            json.dump(
                data,
                f,
                indent=2
            )

    except Exception:
        pass


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


# =========================================================
# WEAPON CLASS CHECK
# =========================================================

def is_weapon_class(class_name):

    name = str(class_name).lower().strip()

    weapon_names = [
        "weapon",
        "gun",
        "pistol",
        "rifle",
        "firearm",
        "knife",
        "sword",
        "shotgun",
        "handgun"
    ]

    for weapon_name in weapon_names:

        if weapon_name in name:
            return True

    return False


# =========================================================
# DISTANCE FUNCTIONS
# =========================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def distance(box1, box2):

    c1 = box_center(box1)
    c2 = box_center(box2)

    return math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )


# =========================================================
# CONVERT VIDEO TO BROWSER FRIENDLY H264
# =========================================================

def convert_to_browser_video(input_path):

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        return input_path

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_h264.mp4"
    ).name

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )

        if result.returncode == 0 and os.path.exists(output_path):

            return output_path

    except Exception:
        pass

    return input_path


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    # Show sidebar only here
    show_home_sidebar()

    # =====================================================
    # RIGHT SIDE - ONLY PROJECT CONTENT
    # =====================================================

    st.markdown(
        """
        <div class="welcome-title">
            🚨 WeaponGuard AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description-card">

            <div class="description-heading">
                Description
            </div>

            <div class="description-text">
                WeaponGuard AI is an intelligent weapon detection
                system designed to improve security through
                automated image and CCTV video analysis. The system
                uses Artificial Intelligence and YOLO-based object
                detection to identify weapons in uploaded media.
                When a weapon is detected, the system highlights
                the detected object and provides an alert sound.
                By reducing the need for continuous manual
                monitoring, the solution helps security personnel
                identify potential threats quickly and respond
                more effectively. The system supports both image
                and video inputs and provides clear visual results
                for easy understanding.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="next-space"></div>',
        unsafe_allow_html=True
    )

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
        "📷 Upload Image for Weapon Detection"
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
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

    st.success(
        "✅ Image uploaded successfully!"
    )

    if st.button(
        "🔍 Detect Weapon",
        use_container_width=True,
        key="image_detect"
    ):

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        image_array = np.array(
            image
        )

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Run YOLO
        results = model(
            image_bgr,
            conf=0.10,
            verbose=False
        )

        output_image = image_bgr.copy()

        weapon_count = 0
        detected_objects = 0

        # =================================================
        # PROCESS DETECTIONS
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

                detected_objects += 1

                if (
                    is_weapon_class(class_name)
                    and confidence >= confidence_threshold
                ):

                    weapon_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

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
                        0.7,
                        (0, 0, 255),
                        2
                    )

        # =================================================
        # DISPLAY
        # =================================================

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="video-title">📥 Input Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_array,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="video-title">📤 Detected Image</div>',
                unsafe_allow_html=True
            )

            output_rgb = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_rgb,
                use_container_width=True
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
        # SAVE LOG
        # =================================================

        save_detection_data(
            "image",
            weapon_count,
            detected_objects
        )

        # =================================================
        # DOWNLOAD
        # =================================================

        output_rgb_pil = Image.fromarray(
            output_rgb
        )

        image_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        )

        output_rgb_pil.save(
            image_temp.name
        )

        image_temp.close()

        with open(
            image_temp.name,
            "rb"
        ) as image_file:

            st.download_button(
                label="⬇️ Download Detected Image",
                data=image_file.read(),
                file_name="weapon_detection_output.png",
                mime="image/png",
                use_container_width=True
            )


# =========================================================
# VIDEO DETECTION
# =========================================================

def video_detection(
    model,
    confidence_threshold,
    required_frames
):

    st.subheader(
        "🎥 Upload CCTV Video"
    )

    uploaded_video = st.file_uploader(
        "Choose a CCTV video",
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

    st.success(
        "✅ Video uploaded successfully!"
    )

    # =====================================================
    # INPUT VIDEO TEMP FILE
    # =====================================================

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.getvalue()
    )

    input_file.close()

    input_video_path = input_file.name

    # =====================================================
    # SHOW ORIGINAL INPUT VIDEO
    # =====================================================

    st.markdown(
        '<div class="video-title">📥 Input CCTV Video</div>',
        unsafe_allow_html=True
    )

    st.video(
        uploaded_video.getvalue()
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # START DETECTION
    # =====================================================

    if not st.button(
        "🔍 Start Weapon Detection",
        use_container_width=True,
        type="primary",
        key="video_detect"
    ):
        return

    # =====================================================
    # OPEN VIDEO
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
    # OUTPUT VIDEO
    # =====================================================

    raw_output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    raw_output_path = raw_output_file.name

    raw_output_file.close()

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        raw_output_path,
        fourcc,
        fps,
        (width, height)
    )

    # =====================================================
    # VARIABLES
    # =====================================================

    frame_number = 0

    last_boxes = {}

    consecutive_weapon_frames = {}

    next_weapon_id = 0

    confirmed_detections = 0

    detection_events = []

    weapon_found_in_video = False

    # =====================================================
    # UI
    # =====================================================

    progress_bar = st.progress(0)

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
            conf=confidence_threshold,
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

            best_id = None
            best_distance = 120

            for prev_id, prev_box in last_boxes.items():

                if prev_id in used_previous_ids:
                    continue

                current_distance = distance(
                    prev_box,
                    weapon_box
                )

                if current_distance < best_distance:

                    best_distance = current_distance
                    best_id = prev_id

            # =============================================
            # EXISTING WEAPON
            # =============================================

            if best_id is not None:

                used_previous_ids.add(
                    best_id
                )

                consecutive_weapon_frames[
                    best_id
                ] = (
                    consecutive_weapon_frames.get(
                        best_id,
                        0
                    ) + 1
                )

                new_last_boxes[
                    best_id
                ] = weapon_box

                if (
                    consecutive_weapon_frames[
                        best_id
                    ] >= required_frames
                ):

                    confirmed_weapon_boxes.append(
                        {
                            "box": weapon_box,
                            "confidence": weapon_conf,
                            "id": best_id
                        }
                    )

            # =============================================
            # NEW WEAPON
            # =============================================

            else:

                new_id = next_weapon_id

                next_weapon_id += 1

                consecutive_weapon_frames[
                    new_id
                ] = 1

                new_last_boxes[
                    new_id
                ] = weapon_box

        last_boxes = new_last_boxes

        # =================================================
        # DRAW DETECTION
        # =================================================

        if confirmed_weapon_boxes:

            weapon_found_in_video = True

            confirmed_detections += len(
                confirmed_weapon_boxes
            )

            for detection in confirmed_weapon_boxes:

                x1, y1, x2, y2 = detection["box"]

                confidence = detection[
                    "confidence"
                ]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    f"WEAPON {confidence:.2f}",
                    (
                        x1,
                        max(y1 - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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

            cv2.putText(
                frame,
                f"!!! {len(confirmed_weapon_boxes)} WEAPON(S) DETECTED !!!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        else:

            cv2.putText(
                frame,
                "No Weapon Detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # =================================================
        # WRITE FRAME
        # =================================================

        out.write(frame)

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
            use_container_width=True
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

    progress_bar.progress(1.0)

    status_text.success(
        "✅ Video processing completed!"
    )

    preview_placeholder.empty()

    # =====================================================
    # CONVERT TO H264
    # =====================================================

    browser_video_path = convert_to_browser_video(
        raw_output_path
    )

    # =====================================================
    # SAVE LOG
    # =====================================================

    save_detection_data(
        "video",
        confirmed_detections,
        total_frames
    )

    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📊 Detection Results"
    )

    col1, col2, col3 = st.columns(3)

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
                <h3>🚨 Confirmations</h3>
                <h2>{confirmed_detections}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        if weapon_found_in_video:

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
    # SIREN ONLY FOR VIDEO
    # =====================================================

    if weapon_found_in_video:

        alert_path = "alert.mp3"

        if os.path.exists(alert_path):

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            st.warning(
                "🚨 Weapon detected in CCTV video!"
            )

            with open(
                alert_path,
                "rb"
            ) as audio_file:

                st.audio(
                    audio_file.read(),
                    format="audio/mp3"
                )

    # =====================================================
    # DETECTION EVENTS
    # =====================================================

    if detection_events:

        st.markdown("---")

        st.subheader(
            "🚨 Detection Events"
        )

        # Show only unique/limited events
        shown_events = detection_events[:30]

        for event in shown_events:

            st.write(
                f"⏱️ Time: "
                f"{event['time']} sec  |  "
                f"Confidence: "
                f"{event['confidence']}"
            )

    # =====================================================
    # OUTPUT VIDEO
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🎥 Processed CCTV Video"
    )

    if os.path.exists(
        browser_video_path
    ):

        with open(
            browser_video_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        # =================================================
        # IMPORTANT:
        # st.video automatically plays browser-supported
        # H264 MP4 video.
        # =================================================

        st.video(
            video_bytes
        )

        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="weapon_detection_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    else:

        st.error(
            "❌ Processed video could not be created."
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

    if (
        browser_video_path != raw_output_path
        and os.path.exists(raw_output_path)
    ):

        try:
            os.remove(
                raw_output_path
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

    confidence_threshold, required_frames = \
        show_detection_sidebar()

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

        return

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
