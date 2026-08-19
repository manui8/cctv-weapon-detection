import streamlit as st
import cv2
import os
import math
import tempfile
import numpy as np

from ultralytics import YOLO
from PIL import Image


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
    "S Bhavya Sri",
    "S Manasa",
    "S Anusha"
]

ALERT_FILE = "alert.mp3"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .college-title {
        text-align: center;
        font-size: 26px;
        font-weight: 600;
        color: #4da6ff;
        margin-bottom: 30px;
    }

    .guide-card {
        background-color: #1b1f2a;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 20px;
    }

    .guide-label {
        color: #aaaaaa;
        font-size: 18px;
    }

    .guide-name {
        color: white;
        font-size: 25px;
        font-weight: bold;
    }

    .team-card {
        background-color: #1b1f2a;
        padding: 22px;
        border-radius: 14px;
        margin-bottom: 25px;
    }

    .team-heading {
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        margin-bottom: 18px;
    }

    .team-lead {
        text-align: center;
        color: #50fa7b;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 12px;
    }

    .team-member {
        text-align: center;
        color: #dddddd;
        font-size: 18px;
        margin: 7px;
    }

    .project-heading {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    .description-heading {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .description-text {
        text-align: center;
        font-size: 18px;
        line-height: 1.7;
        color: #dddddd;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 30px;
    }

    .detection-box {
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 23px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .danger {
        background-color: #4a1111;
        color: #ff5555;
    }

    .safe {
        background-color: #123d24;
        color: #50fa7b;
    }

    .video-heading {
        text-align: center;
        font-size: 23px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .sidebar-heading {
        text-align: center;
        font-size: 21px;
        font-weight: bold;
        padding: 8px;
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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# =========================================================
# HELPER FUNCTIONS
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


def play_alert():

    if os.path.exists(ALERT_FILE):

        with open(ALERT_FILE, "rb") as audio_file:

            audio_bytes = audio_file.read()

        st.audio(
            audio_bytes,
            format="audio/mp3",
            autoplay=True
        )

    else:

        st.warning(
            "⚠️ alert.mp3 file not found."
        )


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-heading">'
            'Artificial Intelligence Career for Women (AICW)'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown("### 🎓 Institution")

        st.write(
            "VSM College of Engineering"
        )

        st.markdown("---")

        st.markdown("### 👨‍🏫 Project Guide")

        st.write(
            "Mr. Abdul Aziz MD"
        )

        st.markdown("---")

        st.markdown("### 👥 Team Members")

        st.write(
            "⭐ **S Nagasindhu — Team Lead**"
        )

        st.write(
            "• S Bhavya Sri — Team Member"
        )

        st.write(
            "• S Manasa — Team Member"
        )

        st.write(
            "• S Anusha — Team Member"
        )

    # -----------------------------------------------------
    # MAIN WELCOME CONTENT
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        'Artificial Intelligence Career for Women (AICW)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="college-title">'
        'VSM College of Engineering'
        '</div>',
        unsafe_allow_html=True
    )

    # Guide

    st.markdown(
        '<div class="guide-card">'
        '<div class="guide-label">Project Guide</div>'
        '<div class="guide-name">Mr. Abdul Aziz MD</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Team

    st.markdown(
        '<div class="team-card">'
        '<div class="team-heading">👥 Team Members</div>'
        '<div class="team-lead">'
        '⭐ S Nagasindhu — Team Lead'
        '</div>'
        '<div class="team-member">'
        'S Bhavya Sri — Team Member'
        '</div>'
        '<div class="team-member">'
        'S Manasa — Team Member'
        '</div>'
        '<div class="team-member">'
        'S Anusha — Team Member'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Project title

    st.markdown(
        '<div class="project-heading">'
        '🚨 WeaponGuard AI'
        '</div>',
        unsafe_allow_html=True
    )

    # Description heading

    st.markdown(
        '<div class="description-heading">'
        'Description'
        '</div>',
        unsafe_allow_html=True
    )

    # Description text

    st.markdown(
        '<div class="description-text">'
        'WeaponGuard AI is an intelligent weapon detection system '
        'designed to identify weapons from CCTV images and video '
        'footage using Artificial Intelligence and YOLO-based object '
        'detection. The system analyzes uploaded media, detects '
        'weapons, highlights the detected objects with bounding boxes, '
        'and provides an alert when a weapon is identified. By '
        'supporting both image and video input, the system improves '
        'security monitoring and helps users respond quickly to '
        'potentially dangerous situations.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Next button

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

def image_detection(model, confidence_threshold):

    st.subheader(
        "📷 Image Weapon Detection"
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
        "✅ Image uploaded successfully!"
    )

    if st.button(
        "🔍 Detect Weapon",
        use_container_width=True,
        key="detect_image"
    ):

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        results = model(
            image_bgr,
            conf=0.10,
            verbose=False
        )

        output_image = image_bgr.copy()

        weapon_count = 0

        # ---------------------------------------------
        # DETECTION
        # ---------------------------------------------

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
                    class_name.lower() == "weapon"
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

                    cv2.putText(
                        output_image,
                        f"WEAPON {confidence:.2f}",
                        (x1, max(y1 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        output_rgb = cv2.cvtColor(
            output_image,
            cv2.COLOR_BGR2RGB
        )

        # ---------------------------------------------
        # SIDE BY SIDE
        # ---------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="video-heading">'
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
                '<div class="video-heading">'
                '📤 Output Image'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                output_rgb,
                use_container_width=True
            )

        # ---------------------------------------------
        # RESULT
        # ---------------------------------------------

        if weapon_count > 0:

            st.markdown(
                f'<div class="detection-box danger">'
                f'🚨 {weapon_count} WEAPON(S) DETECTED'
                f'</div>',
                unsafe_allow_html=True
            )

            play_alert()

        else:

            st.markdown(
                '<div class="detection-box safe">'
                '✅ NO WEAPON DETECTED'
                '</div>',
                unsafe_allow_html=True
            )


# =========================================================
# VIDEO DETECTION
# =========================================================

def video_detection(model, confidence_threshold, required_frames):

    st.subheader(
        "🎥 CCTV Video Weapon Detection"
    )

    uploaded_video = st.file_uploader(
        "Upload CCTV Video",
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

    # -----------------------------------------------------
    # SAVE INPUT VIDEO
    # -----------------------------------------------------

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.getbuffer()
    )

    input_file.close()

    input_video_path = input_file.name

    # -----------------------------------------------------
    # SHOW ORIGINAL INPUT VIDEO
    # -----------------------------------------------------

    st.markdown(
        '<div class="video-heading">'
        '📥 Input CCTV Video'
        '</div>',
        unsafe_allow_html=True
    )

    # Original video playable directly

    st.video(
        uploaded_video
    )

    st.markdown("---")

    # -----------------------------------------------------
    # START DETECTION
    # -----------------------------------------------------

    start_detection = st.button(
        "🔍 START WEAPON DETECTION",
        use_container_width=True,
        type="primary",
        key="start_video_detection"
    )

    if not start_detection:
        return

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        input_video_path
    )

    if not cap.isOpened():

        st.error(
            "❌ Unable to open video."
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

    # -----------------------------------------------------
    # TEMP OUTPUT
    # -----------------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_video_path = output_file.name

    output_file.close()

    # -----------------------------------------------------
    # VIDEO WRITER
    # -----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    # -----------------------------------------------------
    # VARIABLES
    # -----------------------------------------------------

    frame_number = 0

    weapon_seen_frames = 0

    weapon_detected = False

    detection_count = 0

    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()

    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        current_weapon = False

        current_confidence = 0

        current_box = None

        # ---------------------------------------------
        # YOLO
        # ---------------------------------------------

        results = model(
            frame,
            conf=0.10,
            verbose=False
        )

        # ---------------------------------------------
        # FIND WEAPON
        # ---------------------------------------------

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
                    class_name.lower() == "weapon"
                    and confidence >= confidence_threshold
                ):

                    current_weapon = True

                    current_confidence = confidence

                    current_box = tuple(
                        map(
                            int,
                            box.xyxy[0]
                        )
                    )

                    break

            if current_weapon:
                break

        # ---------------------------------------------
        # CONSECUTIVE DETECTION
        # ---------------------------------------------

        if current_weapon:

            weapon_seen_frames += 1

        else:

            weapon_seen_frames = 0

        if (
            weapon_seen_frames >= required_frames
        ):

            weapon_detected = True

        # ---------------------------------------------
        # DRAW OUTPUT
        # ---------------------------------------------

        if current_weapon and current_box:

            x1, y1, x2, y2 = current_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"WEAPON {current_confidence:.2f}",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "!!! WEAPON DETECTED !!!",
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

        # ---------------------------------------------
        # WRITE OUTPUT FRAME
        # ---------------------------------------------

        out.write(frame)

        # ---------------------------------------------
        # PROGRESS
        # ---------------------------------------------

        if total_frames > 0:

            progress = (
                frame_number /
                total_frames
            )

            progress_bar.progress(
                min(progress, 1.0)
            )

        status_text.write(
            f"Processing video: "
            f"{frame_number} / "
            f"{total_frames} frames"
        )

    # -----------------------------------------------------
    # RELEASE
    # -----------------------------------------------------

    cap.release()

    out.release()

    progress_bar.progress(1.0)

    status_text.success(
        "✅ Weapon detection completed!"
    )

    # -----------------------------------------------------
    # OUTPUT VIDEO
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="video-heading">'
        '📤 Output CCTV Video'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # OUTPUT SIDE BY SIDE
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="video-heading">'
            '📥 Input Video'
            '</div>',
            unsafe_allow_html=True
        )

        # Play original uploaded video

        with open(
            input_video_path,
            "rb"
        ) as f:

            input_bytes = f.read()

        st.video(
            input_bytes
        )

    with col2:

        st.markdown(
            '<div class="video-heading">'
            '📤 Detected Output Video'
            '</div>',
            unsafe_allow_html=True
        )

        # Read processed video

        if os.path.exists(
            output_video_path
        ):

            with open(
                output_video_path,
                "rb"
            ) as f:

                output_bytes = f.read()

            st.video(
                output_bytes
            )

    # -----------------------------------------------------
    # DETECTION RESULT
    # -----------------------------------------------------

    st.markdown("---")

    if weapon_detected:

        st.markdown(
            '<div class="detection-box danger">'
            '🚨 WEAPON DETECTED IN CCTV VIDEO'
            '</div>',
            unsafe_allow_html=True
        )

        # Alert only after processing confirms weapon

        play_alert()

    else:

        st.markdown(
            '<div class="detection-box safe">'
            '✅ NO WEAPON DETECTED'
            '</div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # DOWNLOAD OPTIONAL
    # -----------------------------------------------------

    if os.path.exists(
        output_video_path
    ):

        with open(
            output_video_path,
            "rb"
        ) as f:

            output_bytes = f.read()

        st.download_button(
            "⬇️ Download Output Video",
            data=output_bytes,
            file_name="weapon_detection_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )


# =========================================================
# DETECTION PAGE
# =========================================================

def detection_page():

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-heading">'
            '⚙️ Detection Settings'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        if st.button(
            "⬅️ BACK",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            0.10,
            1.00,
            CONFIDENCE_THRESHOLD,
            0.05
        )

        required_frames = st.slider(
            "🎞️ Required Consecutive Frames",
            1,
            15,
            REQUIRED_CONSECUTIVE_FRAMES,
            1
        )

        st.markdown("---")

        st.write(
            "🤖 **Model:** YOLO"
        )

        st.write(
            "🎯 **Detection:** Weapon"
        )

        st.write(
            "📷 **Input:** Image / CCTV Video"
        )

    # -----------------------------------------------------
    # MODEL CHECK
    # -----------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ best.pt not found!"
        )

        st.info(
            "Place best.pt in the same folder "
            "as app.py."
        )

        st.stop()

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="project-heading">'
        '🚨 WeaponGuard AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="description-text">'
        'AI-powered weapon detection from images and CCTV video.'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------

    image_tab, video_tab = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection"
        ]
    )

    with image_tab:

        image_detection(
            model,
            confidence_threshold
        )

    with video_tab:

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
