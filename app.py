import streamlit as st
import cv2
import os
import math
import tempfile
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

    /* ==============================
       GENERAL
       ============================== */

    .stApp {
        background-color: #0e1117;
    }

    .main {
        background-color: #0e1117;
    }

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #151922;
    }

    .aicw-sidebar-title {
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        color: white;
        line-height: 1.35;
        padding: 15px 5px 20px 5px;
    }

    .sidebar-section-title {
        color: #4da6ff;
        font-size: 18px;
        font-weight: 700;
        margin-top: 8px;
    }

    .sidebar-text {
        color: #eeeeee;
        font-size: 16px;
        line-height: 1.6;
    }

    .sidebar-team {
        color: #eeeeee;
        font-size: 15px;
        line-height: 1.8;
    }

    /* ==============================
       HOME PAGE
       ============================== */

    .home-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: white;
        margin-top: 35px;
        margin-bottom: 25px;
    }

    .description-heading {
        text-align: center;
        font-size: 27px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .description-text {
        background-color: #181d27;
        border-radius: 14px;
        padding: 25px 32px;
        color: #dddddd;
        font-size: 18px;
        line-height: 1.8;
        text-align: justify;
        margin-bottom: 30px;
    }

    .home-info {
        text-align: center;
        color: #aeb5c2;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* ==============================
       PROJECT PAGE
       ============================== */

    .project-page-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .project-page-subtitle {
        text-align: center;
        color: #aeb5c2;
        font-size: 18px;
        margin-bottom: 25px;
    }

    /* ==============================
       DETECTION RESULT
       ============================== */

    .detection-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 23px;
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
        background-color: #181d27;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
    }

    .video-title {
        text-align: center;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* ==============================
       BUTTON
       ============================== */

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
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


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
# BOX CENTER
# =========================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


# =========================================================
# DISTANCE
# =========================================================

def distance(box1, box2):

    c1 = box_center(box1)
    c2 = box_center(box2)

    return math.sqrt(
        (c1[0] - c2[0]) ** 2
        +
        (c1[1] - c2[1]) ** 2
    )


# =========================================================
# ALERT SOUND
# =========================================================

def show_alert_sound():

    alert_path = "alert.mp3"

    if os.path.exists(alert_path):

        try:

            with open(
                alert_path,
                "rb"
            ) as audio_file:

                st.audio(
                    audio_file.read(),
                    format="audio/mp3"
                )

        except Exception:
            pass


# =========================================================
# SIDEBAR - HOME
# =========================================================

def home_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div class="aicw-sidebar-title">
                Artificial Intelligence<br>
                Career for Women (AICW)
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
            '<div class="sidebar-text">'
            'VSM College of Engineering'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">'
            '👨‍🏫 Project Guide'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-text">'
            'Mr. Abdul Aziz MD'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">'
            '👥 Team Members'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sidebar-team">
                ⭐ <b>{TEAM_LEAD}</b> — Team Lead<br>
                • {TEAM_MEMBERS[0]} — Team Member<br>
                • {TEAM_MEMBERS[1]} — Team Member<br>
                • {TEAM_MEMBERS[2]} — Team Member
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.caption(
            "WeaponGuard AI"
        )


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    home_sidebar()

    # -----------------------------------------------
    # MAIN TITLE
    # -----------------------------------------------

    st.markdown(
        """
        <div class="home-title">
            🚨 WeaponGuard AI
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------
    # DESCRIPTION HEADING
    # -----------------------------------------------

    st.markdown(
        """
        <div class="description-heading">
            Description
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------
    # DESCRIPTION
    # -----------------------------------------------

    st.markdown(
        """
        <div class="description-text">

        <b>WeaponGuard AI</b> is an AI-powered CCTV weapon
        detection system designed to improve public safety and
        security. The system uses the YOLO deep learning model
        to analyze uploaded images and CCTV video footage and
        automatically identify weapons. When a weapon is detected,
        the system highlights the detected object with a bounding
        box and displays the detection result with confidence.
        It also provides an alert sound to indicate a potential
        security threat. By reducing the need for continuous
        manual monitoring, WeaponGuard AI helps security personnel
        respond quickly and take appropriate action during
        potentially dangerous situations.

        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------
    # SMALL INFO
    # -----------------------------------------------

    st.markdown(
        """
        <div class="home-info">
            AI-powered weapon detection using images and CCTV videos
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------
    # NEXT BUTTON
    # -----------------------------------------------

    if st.button(
        "➡️ NEXT",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.page = "detection"

        st.rerun()


# =========================================================
# DETECTION SIDEBAR
# =========================================================

def detection_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div class="aicw-sidebar-title">
                Artificial Intelligence<br>
                Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # -------------------------------------------
        # BACK BUTTON
        # -------------------------------------------

        if st.button(
            "⬅️ BACK",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        # -------------------------------------------
        # SETTINGS
        # -------------------------------------------

        st.markdown(
            '<div class="sidebar-section-title">'
            '⚙️ Detection Settings'
            '</div>',
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

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-text">'
            '🤖 <b>Model:</b> YOLO<br>'
            '🎯 <b>Detection:</b> Weapon<br>'
            '📷 <b>Input:</b> Image / CCTV Video'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">'
            '👥 Team'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="sidebar-team">
                ⭐ <b>{TEAM_LEAD}</b> — Team Lead<br>
                • {TEAM_MEMBERS[0]} — Team Member<br>
                • {TEAM_MEMBERS[1]} — Team Member<br>
                • {TEAM_MEMBERS[2]} — Team Member
            </div>
            """,
            unsafe_allow_html=True
        )

    return confidence_threshold, required_frames


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

        st.info(
            "Upload an image to start weapon detection."
        )

        return

    if st.button(
        "🔍 Detect Weapon",
        use_container_width=True,
        key="image_detect_button"
    ):

        # -------------------------------------------
        # READ IMAGE
        # -------------------------------------------

        image = Image.open(
            uploaded_image
        )

        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        image_array = np.array(
            image
        )

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # -------------------------------------------
        # YOLO DETECTION
        # -------------------------------------------

        results = model(
            image_bgr,
            conf=0.10,
            verbose=False
        )

        output_image = image_bgr.copy()

        weapon_count = 0
        detected_objects = 0

        # -------------------------------------------
        # PROCESS DETECTIONS
        # -------------------------------------------

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

                class_name = str(
                    model.names[class_id]
                )

                detected_objects += 1

                if (
                    class_name.lower() == "weapon"
                    and
                    confidence >= confidence_threshold
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

        # -------------------------------------------
        # SIDE BY SIDE
        # -------------------------------------------

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="video-title">'
                '📥 Input Image'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_array,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="video-title">'
                '📤 Output Image'
                '</div>',
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

        # -------------------------------------------
        # SAVE LOG
        # -------------------------------------------

        save_detection_data(
            "image",
            weapon_count,
            detected_objects
        )

        # -------------------------------------------
        # RESULT
        # -------------------------------------------

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

        st.info(
            "Upload a CCTV video to start detection."
        )

        return

    # -----------------------------------------------
    # START DETECTION
    # -----------------------------------------------

    if st.button(
        "🔍 Start Weapon Detection",
        use_container_width=True,
        type="primary",
        key="start_video_detection"
    ):

        # -------------------------------------------
        # SAVE INPUT VIDEO TEMPORARILY
        # -------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.getvalue()
        )

        input_file.close()

        input_video_path = (
            input_file.name
        )

        # -------------------------------------------
        # OUTPUT VIDEO
        # -------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_video_path = (
            output_file.name
        )

        output_file.close()

        # -------------------------------------------
        # OPEN INPUT
        # -------------------------------------------

        cap = cv2.VideoCapture(
            input_video_path
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open uploaded video."
            )

            try:
                os.remove(input_video_path)
            except Exception:
                pass

            return

        # -------------------------------------------
        # VIDEO INFORMATION
        # -------------------------------------------

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

        # -------------------------------------------
        # VIDEO WRITER
        # -------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )

        # -------------------------------------------
        # TRACKING VARIABLES
        # -------------------------------------------

        frame_number = 0

        consecutive_weapon_frames = {}

        last_boxes = {}

        confirmed_detections = 0

        detection_events = []

        # -------------------------------------------
        # PROGRESS UI
        # -------------------------------------------

        progress_bar = st.progress(0)

        status_text = st.empty()

        preview_title = st.empty()

        preview_image = st.empty()

        # -------------------------------------------
        # PROCESS VIDEO
        # -------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            current_weapons = []

            # ---------------------------------------
            # YOLO
            # ---------------------------------------

            results = model(
                frame,
                conf=confidence_threshold,
                verbose=False
            )

            # ---------------------------------------
            # FIND WEAPONS
            # ---------------------------------------

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

                    class_name = str(
                        model.names[class_id]
                    )

                    if (
                        class_name.lower()
                        == "weapon"
                        and
                        confidence
                        >= confidence_threshold
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
                                "confidence":
                                    confidence
                            }
                        )

            # ---------------------------------------
            # TRACK WEAPONS
            # ---------------------------------------

            confirmed_weapon_boxes = []

            new_last_boxes = {}

            used_previous_ids = set()

            for weapon in current_weapons:

                weapon_box = weapon["box"]

                weapon_confidence = (
                    weapon["confidence"]
                )

                best_match_id = None

                best_match_distance = float("inf")

                for (
                    previous_id,
                    previous_box
                ) in last_boxes.items():

                    if previous_id in used_previous_ids:
                        continue

                    current_distance = distance(
                        previous_box,
                        weapon_box
                    )

                    if (
                        current_distance
                        < best_match_distance
                    ):

                        best_match_distance = (
                            current_distance
                        )

                        best_match_id = (
                            previous_id
                        )

                # -----------------------------------
                # EXISTING WEAPON
                # -----------------------------------

                if (
                    best_match_id is not None
                    and
                    best_match_distance < 150
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
                        ]
                        >= required_frames
                    ):

                        confirmed_weapon_boxes.append(
                            {
                                "box":
                                    weapon_box,
                                "confidence":
                                    weapon_confidence,
                                "id":
                                    best_match_id
                            }
                        )

                # -----------------------------------
                # NEW WEAPON
                # -----------------------------------

                else:

                    existing_ids = (
                        list(
                            consecutive_weapon_frames.keys()
                        )
                    )

                    if existing_ids:

                        new_id = (
                            max(existing_ids) + 1
                        )

                    else:

                        new_id = 0

                    consecutive_weapon_frames[
                        new_id
                    ] = 1

                    new_last_boxes[
                        new_id
                    ] = weapon_box

            # ---------------------------------------
            # UPDATE TRACKING
            # ---------------------------------------

            last_boxes = new_last_boxes

            # ---------------------------------------
            # DRAW DETECTIONS
            # ---------------------------------------

            if confirmed_weapon_boxes:

                confirmed_detections += len(
                    confirmed_weapon_boxes
                )

                for detection in confirmed_weapon_boxes:

                    x1, y1, x2, y2 = (
                        detection["box"]
                    )

                    confidence = (
                        detection["confidence"]
                    )

                    # Bounding box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    # Label

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
                            "time":
                                round(
                                    time_sec,
                                    2
                                ),
                            "confidence":
                                round(
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

            # ---------------------------------------
            # WRITE OUTPUT FRAME
            # ---------------------------------------

            out.write(frame)

            # ---------------------------------------
            # LIVE PREVIEW
            # ---------------------------------------

            preview_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            preview_title.markdown(
                "🔄 Processing CCTV Video..."
            )

            preview_image.image(
                preview_rgb,
                use_container_width=True
            )

            # ---------------------------------------
            # PROGRESS
            # ---------------------------------------

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

        # -------------------------------------------
        # RELEASE VIDEO
        # -------------------------------------------

        cap.release()

        out.release()

        progress_bar.progress(1.0)

        status_text.success(
            "✅ Video processing completed!"
        )

        preview_title.empty()
        preview_image.empty()

        # -------------------------------------------
        # SAVE LOG
        # -------------------------------------------

        save_detection_data(
            "video",
            confirmed_detections,
            total_frames
        )

        # -------------------------------------------
        # RESULT SUMMARY
        # -------------------------------------------

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

            if confirmed_detections > 0:

                st.markdown(
                    """
                    <div class="detection-box danger">
                        🚨 WEAPON DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

        # -------------------------------------------
        # DETECTION EVENTS
        # -------------------------------------------

        if detection_events:

            st.markdown("---")

            st.subheader(
                "🚨 Detection Events"
            )

            # Show limited events to keep UI clean

            displayed_events = (
                detection_events[:30]
            )

            for event in displayed_events:

                st.warning(
                    f"⏱️ Time: "
                    f"{event['time']} sec | "
                    f"Confidence: "
                    f"{event['confidence']}"
                )

            if len(detection_events) > 30:

                st.caption(
                    "Showing first 30 detection events."
                )

        # -------------------------------------------
        # READ OUTPUT VIDEO
        # -------------------------------------------

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

            # ---------------------------------------
            # INPUT + OUTPUT VIDEO SIDE BY SIDE
            # ---------------------------------------

            st.markdown("---")

            st.subheader(
                "🎥 Input & Output Video"
            )

            video_col1, video_col2 = st.columns(2)

            with video_col1:

                st.markdown(
                    '<div class="video-title">'
                    '📥 Input CCTV Video'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.video(
                    uploaded_video.getvalue()
                )

            with video_col2:

                st.markdown(
                    '<div class="video-title">'
                    '📤 Output Video — WeaponGuard AI'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Plays directly in browser.
                # No download required.

                st.video(
                    output_video_bytes
                )

        # -------------------------------------------
        # OPTIONAL DOWNLOAD
        # -------------------------------------------

        st.markdown("---")

        st.download_button(
            label="⬇️ Download Processed Video",
            data=output_video_bytes,
            file_name="weapon_detection_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        # -------------------------------------------
        # CLEAN TEMP INPUT
        # -------------------------------------------

        try:

            os.remove(
                input_video_path
            )

        except Exception:
            pass


# =========================================================
# DETECTION PAGE
# =========================================================

def detection_page():

    confidence_threshold, required_frames = (
        detection_sidebar()
    )

    # -----------------------------------------------
    # CHECK MODEL
    # -----------------------------------------------

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

    # -----------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------

    try:

        model = load_model()

    except Exception as e:

        st.error(
            f"❌ Error loading YOLO model: {e}"
        )

        return

    # -----------------------------------------------
    # HEADER
    # -----------------------------------------------

    st.markdown(
        """
        <div class="project-page-title">
            🚨 WeaponGuard AI
        </div>

        <div class="project-page-subtitle">
            AI-powered weapon detection from images and CCTV videos
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------
    # TABS
    # -----------------------------------------------

    tab1, tab2 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection"
        ]
    )

    # -----------------------------------------------
    # IMAGE TAB
    # -----------------------------------------------

    with tab1:

        image_detection(
            model,
            confidence_threshold
        )

    # -----------------------------------------------
    # VIDEO TAB
    # -----------------------------------------------

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

elif st.session_state.page == "detection":

    detection_page()
