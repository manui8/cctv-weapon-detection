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

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

/* =========================
   WELCOME PAGE
   ========================= */

.home-container {
    text-align: center;
    padding-top: 35px;
}

.aicw-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.college-name {
    font-size: 27px;
    font-weight: 600;
    color: #4da6ff;
    margin-bottom: 35px;
}

.guide-box {
    background-color: #1b1f2a;
    padding: 22px;
    border-radius: 15px;
    margin: 15px auto;
    max-width: 850px;
}

.guide-title {
    font-size: 20px;
    color: #aaaaaa;
    margin-bottom: 5px;
}

.guide-name {
    font-size: 26px;
    font-weight: bold;
    color: #ffffff;
}

.team-box {
    background-color: #1b1f2a;
    padding: 25px;
    border-radius: 15px;
    margin: 20px auto;
    max-width: 850px;
}

.team-title {
    font-size: 26px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 20px;
}

.team-lead {
    font-size: 21px;
    color: #50fa7b;
    font-weight: bold;
    margin-bottom: 15px;
}

.team-member {
    font-size: 19px;
    color: #dddddd;
    margin: 8px;
}

.project-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 45px;
    margin-bottom: 25px;
    text-align: center;
}

.project-subtitle {
    font-size: 19px;
    color: #aaaaaa;
    text-align: center;
    margin-bottom: 30px;
}

/* =========================
   DETECTION PAGE
   ========================= */

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}

.detection-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
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

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# HOME / WELCOME PAGE
# =========================================================

def home_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-title">'
            '📌 PROJECT INFORMATION'
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
            "• S Bhavyasri"
        )

        st.write(
            "• S Manasa"
        )

        st.write(
            "• S Anusha"
        )

        st.markdown("---")

        st.info(
            "AI Based Weapon Detection in CCTV"
        )

    # =====================================================
    # MAIN WELCOME CONTENT
    # =====================================================

    st.markdown(
        '<div class="home-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="aicw-title">'
        'Artificial Intelligence Career for Women (AICW)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="college-name">'
        'VSM College of Engineering'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # PROJECT GUIDE
    # =====================================================

    st.markdown(
        """
        <div class="guide-box">

            <div class="guide-title">
                Project Guide
            </div>

            <div class="guide-name">
                Mr. Abdul Aziz MD
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # TEAM MEMBERS
    # =====================================================

    st.markdown(
        """
        <div class="team-box">

            <div class="team-title">
                👥 Team Members
            </div>

            <div class="team-lead">
                ⭐ S Nagasindhu — Team Lead
            </div>

            <div class="team-member">
                S Bhavyasri — Team Member
            </div>

            <div class="team-member">
                S Manasa — Team Member
            </div>

            <div class="team-member">
                S Anusha — Team Member
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # PROJECT TITLE
    # =====================================================

    st.markdown(
        '<div class="project-title">'
        '🚨 AI Based Weapon Detection in CCTV'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="project-subtitle">'
        'An AI-powered system for automatic weapon detection '
        'from images and CCTV video.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # =====================================================
    # NEXT BUTTON
    # =====================================================

    if st.button(
        "➡️ NEXT — OPEN WEAPON DETECTION",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.page = "detection"

        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


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
# SAVE DETECTION DATA
# =========================================================

def save_detection_data(
    detection_type,
    weapon_count,
    total_objects=0
):

    data_file = "detection_log.json"

    detection_record = {

        "timestamp":
            datetime.now().isoformat(),

        "type":
            detection_type,

        "weapons_detected":
            weapon_count,

        "total_objects":
            total_objects
    }

    try:

        if os.path.exists(
            data_file
        ):

            with open(
                data_file,
                "r"
            ) as f:

                data = json.load(f)

        else:

            data = {
                "detections": []
            }

        data[
            "detections"
        ].append(
            detection_record
        )

        if len(
            data["detections"]
        ) > 100:

            data[
                "detections"
            ] = data[
                "detections"
            ][-100:]

        with open(
            data_file,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

    except Exception as e:

        st.warning(
            f"Could not save detection data: {e}"
        )


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(
        MODEL_PATH
    )


# =========================================================
# ALERT SOUND
# =========================================================

def play_alert_sound():

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

        st.warning(
            "⚠️ alert.mp3 not found. "
            "Please upload alert.mp3 to the same "
            "folder as app.py."
        )


# =========================================================
# DETECTION PAGE
# =========================================================

def detection_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-title">'
            '⚙️ DETECTION SETTINGS'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # =================================================
        # BACK BUTTON
        # =================================================

        if st.button(
            "← Back to Welcome",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        # =================================================
        # CONFIDENCE
        # =================================================

        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=CONFIDENCE_THRESHOLD,
            step=0.05
        )

        # =================================================
        # REQUIRED FRAMES
        # =================================================

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
            "📹 **Input:** CCTV Video / Image"
        )

        st.markdown("---")

        st.markdown(
            "### 👥 Team"
        )

        st.write(
            "⭐ **S Nagasindhu — Team Lead**"
        )

        st.write(
            "• S Bhavyasri"
        )

        st.write(
            "• S Manasa"
        )

        st.write(
            "• S Anusha"
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
        '<div class="title">'
        '🚨 AI Based Weapon Detection in CCTV'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered weapon detection using YOLO'
        '</div>',
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

        if uploaded_image is not None:

            st.success(
                "✅ Image uploaded successfully!"
            )

            if st.button(
                "🔍 Detect Weapons in Image",
                use_container_width=True,
                key="image_detect"
            ):

                # =========================================
                # READ IMAGE
                # =========================================

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

                # =========================================
                # YOLO DETECTION
                # =========================================

                results = model(
                    image_bgr,
                    conf=0.1,
                    verbose=False
                )

                output_image = \
                    image_bgr.copy()

                weapon_count = 0

                detected_objects = 0

                # =========================================
                # PROCESS DETECTIONS
                # =========================================

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

                        class_name = \
                            model.names[
                                class_id
                            ]

                        detected_objects += 1

                        # ---------------------------------
                        # WEAPON CHECK
                        # ---------------------------------

                        if (
                            class_name.lower()
                            == "weapon"
                            and
                            confidence
                            >= confidence_threshold
                        ):

                            weapon_count += 1

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0]
                            )

                            # -----------------------------
                            # DRAW BOX
                            # -----------------------------

                            cv2.rectangle(
                                output_image,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                3
                            )

                            # -----------------------------
                            # LABEL
                            # -----------------------------

                            label = (
                                f"WEAPON "
                                f"{confidence:.2f}"
                            )

                            cv2.putText(
                                output_image,
                                label,
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

                # =========================================
                # CONVERT OUTPUT
                # =========================================

                output_rgb = cv2.cvtColor(
                    output_image,
                    cv2.COLOR_BGR2RGB
                )

                # =========================================
                # SIDE BY SIDE
                # =========================================

                st.markdown("---")

                col1, col2 = st.columns(
                    2
                )

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

                    st.image(
                        output_rgb,
                        use_container_width=True
                    )

                # =========================================
                # SAVE LOG
                # =========================================

                save_detection_data(
                    "image",
                    weapon_count,
                    detected_objects
                )

                # =========================================
                # RESULT
                # =========================================

                st.markdown("---")

                st.subheader(
                    "🔍 Detection Results"
                )

                if weapon_count > 0:

                    st.markdown(
                        f"""
                        <div class="detection-box danger">
                        🚨 {weapon_count} WEAPON(S) DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "### 🔊 Alert Sound"
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

            st.success(
                "✅ Video uploaded successfully!"
            )

            # =============================================
            # SAVE INPUT VIDEO
            # =============================================

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(
                uploaded_video.read()
            )

            input_file.close()

            input_video_path = \
                input_file.name

            # =============================================
            # OUTPUT VIDEO
            # =============================================

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_video_path = \
                output_file.name

            output_file.close()

            # =============================================
            # START DETECTION
            # =============================================

            if st.button(
                "🔍 Start Weapon Detection",
                use_container_width=True,
                key="video_detect"
            ):

                cap = cv2.VideoCapture(
                    input_video_path
                )

                if not cap.isOpened():

                    st.error(
                        "❌ Could not open uploaded video."
                    )

                    st.stop()

                # =========================================
                # VIDEO PROPERTIES
                # =========================================

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

                # =========================================
                # VIDEO WRITER
                # =========================================

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                out = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (
                        width,
                        height
                    )
                )

                # =========================================
                # VARIABLES
                # =========================================

                frame_number = 0

                consecutive_weapon_frames = {}

                confirmed_detections = 0

                last_boxes = {}

                detection_events = []

                # =========================================
                # UI
                # =========================================

                progress_bar = st.progress(
                    0
                )

                status_text = st.empty()

                # =========================================
                # PROCESSING PREVIEW
                # =========================================

                processing_placeholder = \
                    st.empty()

                # =========================================
                # PROCESS VIDEO
                # =========================================

                while True:

                    ret, frame = cap.read()

                    if not ret:

                        break

                    frame_number += 1

                    current_weapons = []

                    # =====================================
                    # YOLO
                    # =====================================

                    results = model(
                        frame,
                        conf=confidence_threshold,
                        verbose=False
                    )

                    # =====================================
                    # FIND WEAPONS
                    # =====================================

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

                            class_name = \
                                model.names[
                                    class_id
                                ]

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

                    # =====================================
                    # TRACK WEAPONS
                    # =====================================

                    confirmed_weapon_boxes = []

                    new_last_boxes = {}

                    new_consecutive_frames = {}

                    for weapon in current_weapons:

                        weapon_box = \
                            weapon["box"]

                        weapon_conf = \
                            weapon["confidence"]

                        best_match_id = None

                        best_match_distance = 100

                        # ---------------------------------
                        # MATCH WITH PREVIOUS FRAME
                        # ---------------------------------

                        for (
                            prev_id,
                            prev_box
                        ) in last_boxes.items():

                            movement = distance(
                                prev_box,
                                weapon_box
                            )

                            if movement < \
                                best_match_distance:

                                best_match_distance = \
                                    movement

                                best_match_id = \
                                    prev_id

                        # ---------------------------------
                        # EXISTING WEAPON
                        # ---------------------------------

                        if best_match_id is not None:

                            consecutive_weapon_frames[
                                best_match_id
                            ] = \
                                consecutive_weapon_frames.get(
                                    best_match_id,
                                    0
                                ) + 1

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
                                            weapon_conf,

                                        "id":
                                            best_match_id
                                    }
                                )

                        # ---------------------------------
                        # NEW WEAPON
                        # ---------------------------------

                        else:

                            new_id = max(
                                consecutive_weapon_frames.keys(),
                                default=-1
                            ) + 1

                            consecutive_weapon_frames[
                                new_id
                            ] = 1

                            new_last_boxes[
                                new_id
                            ] = weapon_box

                    # =====================================
                    # UPDATE TRACKING
                    # =====================================

                    last_boxes = \
                        new_last_boxes

                    # =====================================
                    # DRAW DETECTION
                    # =====================================

                    if confirmed_weapon_boxes:

                        confirmed_detections += \
                            len(
                                confirmed_weapon_boxes
                            )

                        for weapon_detection in \
                            confirmed_weapon_boxes:

                            x1, y1, x2, y2 = \
                                weapon_detection[
                                    "box"
                                ]

                            confidence = \
                                weapon_detection[
                                    "confidence"
                                ]

                            # -----------------------------
                            # RED BOX
                            # -----------------------------

                            cv2.rectangle(
                                frame,
                                (
                                    x1,
                                    y1
                                ),
                                (
                                    x2,
                                    y2
                                ),
                                (0, 0, 255),
                                3
                            )

                            # -----------------------------
                            # LABEL
                            # -----------------------------

                            label = (
                                f"WEAPON "
                                f"{confidence:.2f}"
                            )

                            cv2.putText(
                                frame,
                                label,
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

                            # -----------------------------
                            # EVENT
                            # -----------------------------

                            time_sec = \
                                frame_number / fps

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

                        # =================================
                        # WARNING TEXT
                        # =================================

                        cv2.putText(
                            frame,
                            (
                                f"!!! "
                                f"{len(confirmed_weapon_boxes)} "
                                f"WEAPON(S) DETECTED !!!"
                            ),
                            (
                                30,
                                50
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

                    else:

                        # =================================
                        # SAFE TEXT
                        # =================================

                        cv2.putText(
                            frame,
                            "No Weapon Detected",
                            (
                                30,
                                50
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                    # =====================================
                    # WRITE OUTPUT FRAME
                    # =====================================

                    out.write(
                        frame
                    )

                    # =====================================
                    # LIVE PROCESSING PREVIEW
                    # =====================================

                    preview_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    processing_placeholder.image(
                        preview_rgb,
                        caption="🔄 Processing CCTV Video...",
                        use_container_width=True
                    )

                    # =====================================
                    # PROGRESS
                    # =====================================

                    if total_frames > 0:

                        progress = (
                            frame_number /
                            total_frames
                        )

                        progress_bar.progress(
                            min(
                                progress,
                                1.0
                            )
                        )

                    status_text.write(
                        f"Processing frame "
                        f"{frame_number} / "
                        f"{total_frames}"
                    )

                # =========================================
                # RELEASE VIDEO
                # =========================================

                cap.release()

                out.release()

                progress_bar.progress(
                    1.0
                )

                status_text.success(
                    "✅ Video processing completed!"
                )

                # Remove processing preview

                processing_placeholder.empty()

                # =========================================
                # SAVE LOG
                # =========================================

                save_detection_data(
                    "video",
                    confirmed_detections,
                    total_frames
                )

                # =========================================
                # RESULTS
                # =========================================

                st.markdown("---")

                st.subheader(
                    "📊 Detection Results"
                )

                col1, col2, col3 = st.columns(
                    3
                )

                # -----------------------------------------
                # FRAMES
                # -----------------------------------------

                with col1:

                    st.markdown(
                        f"""
                        <div class="info-card">

                            <h3>
                                🎞️ Frames
                            </h3>

                            <h2>
                                {total_frames}
                            </h2>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # -----------------------------------------
                # CONFIRMATIONS
                # -----------------------------------------

                with col2:

                    st.markdown(
                        f"""
                        <div class="info-card">

                            <h3>
                                🚨 Confirmations
                            </h3>

                            <h2>
                                {confirmed_detections}
                            </h2>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # -----------------------------------------
                # STATUS
                # -----------------------------------------

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

                    else:

                        st.markdown(
                            """
                            <div class="detection-box safe">

                            ✅ NO WEAPON DETECTED

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # =========================================
                # ALERT SOUND
                # =========================================

                if confirmed_detections > 0:

                    st.markdown("---")

                    st.subheader(
                        "🔊 Weapon Detection Alert"
                    )

                    play_alert_sound()

                # =========================================
                # DETECTION EVENTS
                # =========================================

                if detection_events:

                    st.markdown("---")

                    st.subheader(
                        "🚨 Detection Events"
                    )

                    for event in detection_events:

                        st.warning(
                            f"⏱️ Time: "
                            f"{event['time']} sec | "
                            f"Confidence: "
                            f"{event['confidence']}"
                        )

                # =========================================
                # INPUT / OUTPUT VIDEO
                # =========================================

                st.markdown("---")

                st.subheader(
                    "🎥 CCTV Video Comparison"
                )

                video_col1, video_col2 = \
                    st.columns(2)

                # =========================================
                # INPUT VIDEO
                # =========================================

                with video_col1:

                    st.markdown(
                        "### 📥 Input Video"
                    )

                    try:

                        with open(
                            input_video_path,
                            "rb"
                        ) as input_video:

                            input_video_bytes = \
                                input_video.read()

                        st.video(
                            input_video_bytes
                        )

                    except Exception:

                        st.warning(
                            "Input video preview "
                            "could not be loaded."
                        )

                # =========================================
                # OUTPUT VIDEO
                # =========================================

                with video_col2:

                    st.markdown(
                        "### 📤 Output Video"

                    )

                    if os.path.exists(
                        output_video_path
                    ):

                        with open(
                            output_video_path,
                            "rb"
                        ) as output_video:

                            output_video_bytes = \
                                output_video.read()

                        # Play directly inside app
                        # No download button

                        st.video(
                            output_video_bytes
                        )

                    else:

                        st.warning(
                            "Output video not available."
                        )

                # =========================================
                # CLEAN TEMP INPUT
                # =========================================

                try:

                    os.remove(
                        input_video_path
                    )

                except:

                    pass


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    home_page()

else:

    detection_page()
