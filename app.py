import streamlit as st
import cv2
import os
import math
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
from datetime import datetime


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Based Weapon Detection in CCTV",
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

COLLEGE_NAME = "VSM College of Engineering"

AICW_NAME = "Artificial Intelligence Career for Women (AICW)"

PROJECT_TITLE = "AI Based Weapon Detection in CCTV"

GUIDE_NAME = "Mr. Abdul Aziz MD"

TEAM_LEAD = "S Nagasindhu"

TEAM_MEMBERS = [
    "S Bhavyasri",
    "S Manasa",
    "S Anusha"
]


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# CUSTOM CSS
# IMPORTANT: CSS MUST STAY INSIDE THIS STRING
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

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #151923;
    }

    .sidebar-aicw {
        font-size: 22px;
        font-weight: 800;
        color: white;
        text-align: center;
        line-height: 1.3;
        padding: 12px 5px 18px 5px;
    }

    .sidebar-section-title {
        font-size: 17px;
        font-weight: 700;
        color: #4da6ff;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .sidebar-text {
        font-size: 15px;
        color: #dddddd;
        line-height: 1.6;
    }

    /* =====================================================
       HOME PAGE
       ===================================================== */

    .welcome-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-top: 25px;
        margin-bottom: 8px;
    }

    .welcome-subtitle {
        text-align: center;
        font-size: 18px;
        color: #aaaaaa;
        margin-bottom: 35px;
    }

    .main-project-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-top: 20px;
        margin-bottom: 30px;
    }

    .description-card {
        background-color: #1b1f2a;
        border-radius: 15px;
        padding: 28px 35px;
        margin: 10px auto 30px auto;
        max-width: 1000px;
        border: 1px solid #292f3d;
    }

    .description-heading {
        text-align: center;
        font-size: 27px;
        font-weight: 700;
        color: white;
        margin-bottom: 18px;
    }

    .description-text {
        font-size: 17px;
        line-height: 1.8;
        color: #dddddd;
        text-align: justify;
    }

    .team-card {
        background-color: #1b1f2a;
        border-radius: 15px;
        padding: 22px;
        margin-top: 20px;
    }

    .team-heading {
        font-size: 23px;
        font-weight: 700;
        color: white;
        margin-bottom: 15px;
    }

    .team-lead {
        font-size: 18px;
        color: #50fa7b;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .team-member {
        font-size: 17px;
        color: #dddddd;
        margin-bottom: 8px;
    }

    /* =====================================================
       DETECTION PAGE
       ===================================================== */

    .detection-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: white;
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
        border: 1px solid #292f3d;
    }

    .back-text {
        font-size: 15px;
        color: #aaaaaa;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def is_weapon_class(class_name):
    """
    Supports common class names used in weapon detection models.
    """

    name = str(class_name).lower().strip()

    weapon_names = [
        "weapon",
        "gun",
        "pistol",
        "rifle",
        "firearm",
        "knife",
        "sword"
    ]

    return name in weapon_names


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

            data["detections"] = data["detections"][-100:]

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
# ALERT SOUND
# =========================================================

def play_alert_sound():

    alert_path = "alert.mp3"

    if os.path.exists(alert_path):

        with open(alert_path, "rb") as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/mp3",
                autoplay=True
            )

    else:

        st.warning(
            "⚠️ alert.mp3 not found. "
            "Place alert.mp3 in the same folder as app.py."
        )


# =========================================================
# HOME / WELCOME PAGE
# =========================================================

def home_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-aicw">
                Artificial Intelligence Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">🎓 College</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="sidebar-text">{COLLEGE_NAME}</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">👨‍🏫 Project Guide</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="sidebar-text">{GUIDE_NAME}</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">👥 Team Members</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sidebar-text">
                ⭐ <b>{TEAM_LEAD}</b> — Team Lead<br>
                • {TEAM_MEMBERS[0]} — Team Member<br>
                • {TEAM_MEMBERS[1]} — Team Member<br>
                • {TEAM_MEMBERS[2]} — Team Member
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            f"""
            <div class="sidebar-text">
                🚨 <b>{PROJECT_TITLE}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # MAIN WELCOME INTERFACE
    # =====================================================

    st.markdown(
        """
        <div class="welcome-title">
            Artificial Intelligence Career for Women (AICW)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="welcome-subtitle">
            {COLLEGE_NAME}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # PROJECT MAIN TITLE
    # =====================================================

    st.markdown(
        f"""
        <div class="main-project-title">
            🚨 {PROJECT_TITLE}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # DESCRIPTION
    # =====================================================

    st.markdown(
        """
        <div class="description-card">

            <div class="description-heading">
                Description
            </div>

            <div class="description-text">
                The AI Based Weapon Detection in CCTV system
                is designed to automatically identify weapons
                from CCTV images and video footage using
                Artificial Intelligence and YOLO-based object
                detection. The system analyzes uploaded media,
                detects suspicious weapons, highlights the
                detected objects, and provides an alert when
                a weapon is identified. This solution helps
                improve security and enables faster response
                to potentially dangerous situations.
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
# DETECTION PAGE
# =========================================================

def detection_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        # BACK BUTTON
        if st.button(
            "⬅️ BACK",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        st.markdown(
            """
            <div class="sidebar-aicw">
                AI Based Weapon Detection
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">⚙️ Detection Settings</div>',
            unsafe_allow_html=True
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=CONFIDENCE_THRESHOLD,
            step=0.05
        )

        required_frames = st.slider(
            "Required Consecutive Frames",
            min_value=1,
            max_value=15,
            value=REQUIRED_CONSECUTIVE_FRAMES,
            step=1
        )

        st.markdown("---")

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

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">👥 Team</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sidebar-text">
                ⭐ <b>{TEAM_LEAD}</b> — Team Lead<br>
                • {TEAM_MEMBERS[0]} — Team Member<br>
                • {TEAM_MEMBERS[1]} — Team Member<br>
                • {TEAM_MEMBERS[2]} — Team Member
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not os.path.exists(MODEL_PATH):

        st.error("❌ best.pt not found!")

        st.info(
            "Please place best.pt in the same folder as app.py."
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
        f"""
        <div class="detection-title">
            🚨 {PROJECT_TITLE}
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
    # IMAGE DETECTION
    # =====================================================

    with tab1:

        st.subheader("📷 Upload Image")

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

        if uploaded_image is not None:

            if st.button(
                "🔍 Detect Weapon",
                use_container_width=True,
                type="primary",
                key="image_detect"
            ):

                image = Image.open(
                    uploaded_image
                )

                if image.mode != "RGB":

                    image = image.convert("RGB")

                image_array = np.array(image)

                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # YOLO detection
                results = model(
                    image_bgr,
                    conf=0.10,
                    verbose=False
                )

                output_image = image_bgr.copy()

                weapon_count = 0

                total_objects = 0

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

                        class_name = model.names[class_id]

                        total_objects += 1

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
                                0.7,
                                (0, 0, 255),
                                2
                            )

                # =================================================
                # SIDE BY SIDE IMAGE
                # =================================================

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("📥 Input Image")

                    st.image(
                        image_array,
                        use_container_width=True
                    )

                with col2:

                    st.subheader("📤 Output Image")

                    output_rgb = cv2.cvtColor(
                        output_image,
                        cv2.COLOR_BGR2RGB
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

                    # ALERT SOUND
                    play_alert_sound()

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
    # VIDEO DETECTION
    # =====================================================

    with tab2:

        st.subheader("🎥 Upload CCTV Video")

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

        if uploaded_video is not None:

            # =================================================
            # SAVE INPUT VIDEO
            # =================================================

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(
                uploaded_video.getbuffer()
            )

            input_file.close()

            input_video_path = input_file.name

            # =================================================
            # START DETECTION
            # =================================================

            if st.button(
                "🔍 Start Weapon Detection",
                use_container_width=True,
                type="primary",
                key="video_detect"
            ):

                # =================================================
                # OPEN VIDEO
                # =================================================

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

                # =================================================
                # OUTPUT VIDEO
                # =================================================

                output_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_video_path = output_file.name

                output_file.close()

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                out = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                # =================================================
                # VARIABLES
                # =================================================

                frame_number = 0

                last_boxes = {}

                consecutive_weapon_frames = {}

                confirmed_detections = 0

                detection_events = []

                # =================================================
                # PROCESSING UI
                # =================================================

                progress_bar = st.progress(0)

                status_text = st.empty()

                live_preview = st.empty()

                # =================================================
                # PROCESS VIDEO
                # =================================================

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

                            class_name = model.names[class_id]

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

                        best_match_distance = float("inf")

                        for (
                            prev_id,
                            prev_box
                        ) in last_boxes.items():

                            if prev_id in used_previous_ids:
                                continue

                            movement = distance(
                                prev_box,
                                weapon_box
                            )

                            if movement < best_match_distance:

                                best_match_distance = movement

                                best_match_id = prev_id

                        # =================================================
                        # EXISTING TRACK
                        # =================================================

                        if (
                            best_match_id is not None
                            and best_match_distance < 150
                        ):

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

                        # =================================================
                        # NEW TRACK
                        # =================================================

                        else:

                            new_id = 0

                            if consecutive_weapon_frames:

                                new_id = (
                                    max(
                                        consecutive_weapon_frames.keys()
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

                        confirmed_detections += len(
                            confirmed_weapon_boxes
                        )

                        for weapon_detection in confirmed_weapon_boxes:

                            x1, y1, x2, y2 = (
                                weapon_detection["box"]
                            )

                            confidence = (
                                weapon_detection["confidence"]
                            )

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
                            (
                                f"!!! "
                                f"{len(confirmed_weapon_boxes)} "
                                f"WEAPON(S) DETECTED !!!"
                            ),
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
                    # WRITE OUTPUT FRAME
                    # =================================================

                    out.write(frame)

                    # =================================================
                    # LIVE PROCESSING PREVIEW
                    # =================================================

                    preview_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    live_preview.image(
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

                # =================================================
                # RELEASE
                # =================================================

                cap.release()

                out.release()

                progress_bar.progress(1.0)

                status_text.success(
                    "✅ Video processing completed!"
                )

                # Remove temporary input file
                try:

                    os.remove(
                        input_video_path
                    )

                except Exception:
                    pass

                # =================================================
                # SAVE LOG
                # =================================================

                save_detection_data(
                    "video",
                    confirmed_detections,
                    total_frames
                )

                # =================================================
                # RESULTS
                # =================================================

                st.markdown("---")

                st.subheader("📊 Detection Results")

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

                    if confirmed_detections > 0:

                        st.markdown(
                            """
                            <div class="detection-box danger">
                                🚨 WEAPON DETECTED
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        play_alert_sound()

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
                # VIDEO SIDE BY SIDE
                # =================================================

                st.markdown("---")

                st.subheader("🎥 Input & Output Video")

                video_col1, video_col2 = st.columns(2)

                with video_col1:

                    st.markdown("### 📥 Input Video")

                    # Replay original uploaded video
                    with open(
                        uploaded_video.name
                        if False else input_video_path,
                        "rb"
                    ) as f:
                        input_video_bytes = f.read()

                    # Input file may have been removed above,
                    # so display from uploaded bytes instead.
                    input_video_bytes = uploaded_video.getvalue()

                    st.video(
                        input_video_bytes
                    )

                with video_col2:

                    st.markdown("### 📤 Output Video")

                    if os.path.exists(
                        output_video_path
                    ):

                        with open(
                            output_video_path,
                            "rb"
                        ) as video_file:

                            output_video_bytes = (
                                video_file.read()
                            )

                        # Direct playback.
                        # No download required.
                        st.video(
                            output_video_bytes
                        )

                # =================================================
                # DETECTION EVENTS
                # =================================================

                if detection_events:

                    st.markdown("---")

                    st.subheader(
                        "🚨 Detection Events"
                    )

                    # Show only unique / important events
                    shown_events = detection_events[:50]

                    for event in shown_events:

                        st.warning(
                            f"⏱️ Time: "
                            f"{event['time']} sec   |   "
                            f"Confidence: "
                            f"{event['confidence']}"
                        )

                # =================================================
                # OPTIONAL DOWNLOAD
                # =================================================

                st.markdown("---")

                st.subheader(
                    "⬇️ Output Video"
                )

                if os.path.exists(
                    output_video_path
                ):

                    with open(
                        output_video_path,
                        "rb"
                    ) as video_file:

                        final_video_bytes = (
                            video_file.read()
                        )

                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=final_video_bytes,
                        file_name="weapon_detection_result.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "detection":

    detection_page()
