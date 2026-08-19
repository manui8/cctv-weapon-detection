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

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #151923;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    /* Welcome heading */
    .welcome-title {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .college-title {
        text-align: center;
        font-size: 23px;
        font-weight: 600;
        color: #4da6ff;
        margin-bottom: 30px;
    }

    /* Description box */
    .description-box {
        background-color: #1b1f2a;
        border: 1px solid #303746;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .description-heading {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .description-text {
        font-size: 17px;
        line-height: 1.8;
        color: #dddddd;
        text-align: justify;
    }

    /* Team box */
    .team-box {
        background-color: #1b1f2a;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }

    /* Detection result */
    .danger-box {
        background-color: #4a1111;
        color: #ff5555;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    .safe-box {
        background-color: #123d24;
        color: #50fa7b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
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


# =========================================================
# HOME / WELCOME PAGE
# =========================================================

def home_page():

    # =====================================================
    # LEFT SIDEBAR
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

        st.markdown("### 🎓 College")
        st.write("VSM College of Engineering")

        st.markdown("---")

        st.markdown("### 👨‍🏫 Project Guide")
        st.write("Mr. Abdul Aziz MD")

        st.markdown("---")

        st.markdown("### 👥 Team Members")

        st.write("⭐ **S Nagasindhu — Team Lead**")
        st.write("• **S Bhavyasri — Team Member**")
        st.write("• **S Manasa — Team Member**")
        st.write("• **S Anusha — Team Member**")

        st.markdown("---")

        st.write("🚨 **WeaponGuard AI**")


    # =====================================================
    # RIGHT SIDE - WELCOME INTERFACE
    # =====================================================

    st.markdown(
        '<div class="weapon-home">',
        unsafe_allow_html=True
    )

    # Main Project Heading

    st.markdown(
        """
        <div class="weapon-title">
            🚨 WeaponGuard AI
        </div>
        """,
        unsafe_allow_html=True
    )

    # Description Heading

    st.markdown(
        """
        <div class="description-heading">
            Description
        </div>
        """,
        unsafe_allow_html=True
    )

    # Description Matter

    st.markdown(
        """
        <div class="description-text">
            WeaponGuard AI is an intelligent CCTV weapon detection
            system designed to improve security through automated
            weapon identification. The system uses Artificial
            Intelligence and YOLO-based object detection to analyze
            uploaded images and CCTV video footage. When a weapon is
            detected, the system highlights the suspicious object and
            provides an immediate alert. This helps security personnel
            identify potential threats quickly, reduce monitoring
            effort, and support faster response to dangerous
            situations. The system provides a simple and user-friendly
            interface for both image and video based weapon detection.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # NEXT BUTTON
    # =====================================================

    st.markdown(
        '<div class="next-area">',
        unsafe_allow_html=True
    )

    if st.button(
        "➡️ NEXT",
        use_container_width=True,
        type="primary"
    ):
        st.session_state.page = "detection"
        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # DESCRIPTION
    #
    # IMPORTANT:
    # No HTML is used here.
    # Therefore HTML CODE will NOT appear.
    # -----------------------------------------------------

    st.markdown(
        '<div class="description-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="description-heading">'
        'Description'
        '</div>',
        unsafe_allow_html=True
    )

    description = """
    The AI Based Weapon Detection in CCTV system is designed
    to automatically identify weapons from CCTV images and
    video footage using Artificial Intelligence and YOLO-based
    object detection. The system analyzes uploaded images and
    videos, detects suspicious weapons, highlights the detected
    objects, and provides an alert when a weapon is identified.
    This solution helps improve security, supports faster
    response, and assists in identifying potentially dangerous
    situations at an early stage.
    """

    st.markdown(
        '<div class="description-text">',
        unsafe_allow_html=True
    )

    st.write(description)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # NEXT BUTTON
    # -----------------------------------------------------

    st.markdown("")

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

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown(
            "## ⚙️ Detection Settings"
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

        st.write("🤖 **Model:** YOLO")

        st.write("🎯 **Detection:** Weapon")

        st.write(
            "📹 **Input:** CCTV Image / Video"
        )

        st.markdown("---")

        st.markdown("### 👥 Team")

        st.write(
            "⭐ **S Nagasindhu — Team Lead**"
        )

        st.write(
            "• S Bhavyasri — Team Member"
        )

        st.write(
            "• S Manasa — Team Member"
        )

        st.write(
            "• S Anusha — Team Member"
        )


    # -----------------------------------------------------
    # CHECK MODEL
    # -----------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ best.pt not found!"
        )

        st.info(
            "Please place best.pt in the same "
            "folder as app.py."
        )

        st.stop()


    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = load_model()


    # -----------------------------------------------------
    # DETECTION PAGE TITLE
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🚨 AI Based Weapon Detection in CCTV'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "AI-powered weapon detection using YOLO"
    )


    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------

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

        st.subheader(
            "📷 Upload Image"
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

        if uploaded_image is not None:

            image = Image.open(
                uploaded_image
            ).convert("RGB")

            image_array = np.array(image)

            st.success(
                "✅ Image uploaded successfully!"
            )

            if st.button(
                "🔍 Detect Weapon",
                use_container_width=True,
                key="image_detect"
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

                detected_objects = 0


                # -------------------------------------------------
                # DETECTION
                # -------------------------------------------------

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
                            class_name.lower()
                            == "weapon"
                            and confidence
                            >= confidence_threshold
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
                                (
                                    x1,
                                    max(
                                        y1 - 10,
                                        30
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )


                # -------------------------------------------------
                # SIDE BY SIDE
                # -------------------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "📥 Input Image"
                    )

                    st.image(
                        image_array,
                        use_container_width=True
                    )

                with col2:

                    st.subheader(
                        "📤 Output Image"
                    )

                    output_rgb = cv2.cvtColor(
                        output_image,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        output_rgb,
                        use_container_width=True
                    )


                # -------------------------------------------------
                # SAVE LOG
                # -------------------------------------------------

                save_detection_data(
                    "image",
                    weapon_count,
                    detected_objects
                )


                # -------------------------------------------------
                # RESULT
                # -------------------------------------------------

                st.markdown("---")

                if weapon_count > 0:

                    st.markdown(
                        f"""
                        <div class="danger-box">
                        🚨 {weapon_count} WEAPON(S) DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    alert_path = "alert.mp3"

                    if os.path.exists(alert_path):

                        with open(
                            alert_path,
                            "rb"
                        ) as audio_file:

                            st.audio(
                                audio_file.read(),
                                format="audio/mp3"
                            )

                else:

                    st.markdown(
                        """
                        <div class="safe-box">
                        ✅ NO WEAPON DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # =====================================================
    # VIDEO DETECTION
    # =====================================================

    with tab2:

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

        if uploaded_video is not None:

            video_bytes_original = uploaded_video.getvalue()

            st.success(
                "✅ Video uploaded successfully!"
            )

            if st.button(
                "🔍 Start Weapon Detection",
                use_container_width=True,
                key="video_detect"
            ):

                # ---------------------------------------------
                # INPUT TEMP FILE
                # ---------------------------------------------

                input_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                input_file.write(
                    video_bytes_original
                )

                input_file.close()

                input_video_path = input_file.name


                # ---------------------------------------------
                # OUTPUT TEMP FILE
                # ---------------------------------------------

                output_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_video_path = output_file.name

                output_file.close()


                # ---------------------------------------------
                # OPEN VIDEO
                # ---------------------------------------------

                cap = cv2.VideoCapture(
                    input_video_path
                )

                if not cap.isOpened():

                    st.error(
                        "❌ Could not open uploaded video."
                    )

                    st.stop()


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


                # ---------------------------------------------
                # VIDEO WRITER
                # ---------------------------------------------

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                out = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (width, height)
                )


                # ---------------------------------------------
                # VARIABLES
                # ---------------------------------------------

                frame_number = 0

                consecutive_weapon_frames = {}

                last_boxes = {}

                confirmed_detections = 0

                detection_events = []


                # ---------------------------------------------
                # UI
                # ---------------------------------------------

                progress_bar = st.progress(0)

                status_text = st.empty()

                live_preview = st.empty()


                # ---------------------------------------------
                # PROCESS VIDEO
                # ---------------------------------------------

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    current_weapons = []


                    # -----------------------------------------
                    # YOLO
                    # -----------------------------------------

                    results = model(
                        frame,
                        conf=confidence_threshold,
                        verbose=False
                    )


                    # -----------------------------------------
                    # FIND WEAPONS
                    # -----------------------------------------

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
                                and confidence
                                >= confidence_threshold
                            ):

                                x1, y1, x2, y2 = map(
                                    int,
                                    box.xyxy[0]
                                )

                                current_weapons.append(
                                    {
                                        "box":
                                            (
                                                x1,
                                                y1,
                                                x2,
                                                y2
                                            ),
                                        "confidence":
                                            confidence
                                    }
                                )


                    # -----------------------------------------
                    # TRACKING
                    # -----------------------------------------

                    confirmed_weapon_boxes = []

                    new_last_boxes = {}

                    new_consecutive_frames = {}


                    for weapon in current_weapons:

                        weapon_box = weapon["box"]

                        weapon_conf = weapon[
                            "confidence"
                        ]

                        best_match_id = None

                        best_match_distance = 100


                        for (
                            prev_id,
                            prev_box
                        ) in last_boxes.items():

                            movement = distance(
                                prev_box,
                                weapon_box
                            )

                            if movement < best_match_distance:

                                best_match_distance = movement

                                best_match_id = prev_id


                        if best_match_id is not None:

                            count = (
                                consecutive_weapon_frames
                                .get(
                                    best_match_id,
                                    0
                                )
                                + 1
                            )

                            new_consecutive_frames[
                                best_match_id
                            ] = count

                            new_last_boxes[
                                best_match_id
                            ] = weapon_box

                            if count >= required_frames:

                                confirmed_weapon_boxes.append(
                                    {
                                        "box":
                                            weapon_box,
                                        "confidence":
                                            weapon_conf,
                                        "id":
                                            best_match_id
                                    }
                                )

                        else:

                            new_id = max(
                                consecutive_weapon_frames.keys(),
                                default=-1
                            ) + 1

                            new_consecutive_frames[
                                new_id
                            ] = 1

                            new_last_boxes[
                                new_id
                            ] = weapon_box


                    consecutive_weapon_frames = (
                        new_consecutive_frames
                    )

                    last_boxes = new_last_boxes


                    # -----------------------------------------
                    # DRAW DETECTIONS
                    # -----------------------------------------

                    if confirmed_weapon_boxes:

                        confirmed_detections += len(
                            confirmed_weapon_boxes
                        )

                        for detection in confirmed_weapon_boxes:

                            x1, y1, x2, y2 = detection[
                                "box"
                            ]

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
                                    max(
                                        y1 - 10,
                                        30
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )

                            detection_events.append(
                                {
                                    "time":
                                        round(
                                            frame_number / fps,
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
                                f"{len(confirmed_weapon_boxes)} "
                                f"WEAPON(S) DETECTED"
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


                    # -----------------------------------------
                    # WRITE OUTPUT FRAME
                    # -----------------------------------------

                    out.write(frame)


                    # -----------------------------------------
                    # LIVE PREVIEW
                    # -----------------------------------------

                    preview_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    live_preview.image(
                        preview_rgb,
                        caption="Processing CCTV Video...",
                        use_container_width=True
                    )


                    # -----------------------------------------
                    # PROGRESS
                    # -----------------------------------------

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


                # ---------------------------------------------
                # RELEASE
                # ---------------------------------------------

                cap.release()

                out.release()

                progress_bar.progress(1.0)

                status_text.success(
                    "✅ Video processing completed!"
                )


                # ---------------------------------------------
                # SAVE LOG
                # ---------------------------------------------

                save_detection_data(
                    "video",
                    confirmed_detections,
                    total_frames
                )


                # ---------------------------------------------
                # SIDE-BY-SIDE VIDEO
                # ---------------------------------------------

                st.markdown("---")

                st.subheader(
                    "🎥 Video Comparison"
                )

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


                    video_col1, video_col2 = st.columns(2)


                    with video_col1:

                        st.markdown(
                            "### 📥 Input Video"
                        )

                        st.video(
                            video_bytes_original
                        )


                    with video_col2:

                        st.markdown(
                            "### 📤 Output Video"
                        )

                        st.video(
                            output_video_bytes
                        )


                    # -----------------------------------------
                    # DETECTION RESULT
                    # -----------------------------------------

                    st.markdown("---")

                    if confirmed_detections > 0:

                        st.markdown(
                            f"""
                            <div class="danger-box">
                            🚨 WEAPON DETECTED
                            <br>
                            Total Confirmations:
                            {confirmed_detections}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        alert_path = "alert.mp3"

                        if os.path.exists(
                            alert_path
                        ):

                            with open(
                                alert_path,
                                "rb"
                            ) as audio_file:

                                st.audio(
                                    audio_file.read(),
                                    format="audio/mp3"
                                )

                    else:

                        st.markdown(
                            """
                            <div class="safe-box">
                            ✅ NO WEAPON DETECTED
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # -----------------------------------------
                    # DETECTION EVENTS
                    # -----------------------------------------

                    if detection_events:

                        st.markdown("---")

                        st.subheader(
                            "🚨 Detection Events"
                        )

                        for event in detection_events:

                            st.write(
                                f"⏱️ Time: "
                                f"{event['time']} sec  |  "
                                f"Confidence: "
                                f"{event['confidence']}"
                            )


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    home_page()

else:

    detection_page()
