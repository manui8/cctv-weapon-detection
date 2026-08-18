import streamlit as st
import cv2
import os
import math
import tempfile
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CCTV Weapon Detection",
    page_icon="🚨",
    layout="wide"
)

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "best.pt"

CONFIDENCE_THRESHOLD = 0.60
REQUIRED_CONSECUTIVE_FRAMES = 5

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

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

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🚨 CCTV Weapon Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered real-time weapon detection using YOLO</div>',
    unsafe_allow_html=True
)

# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    st.error("❌ best.pt not found!")

    st.info(
        "Please upload best.pt to the same folder as app.py."
    )

    st.stop()



# =========================================================
# FIRST / WELCOME SCREEN
# =========================================================

import re

# Create session variables
if "project_started" not in st.session_state:
    st.session_state.project_started = False

if "alert_email" not in st.session_state:
    st.session_state.alert_email = ""


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.project_started:

    st.markdown("""
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f7f9fc 0%,
            #eef2f7 100%
        );
    }

    /* Left information panel */
    .info-panel {
        background: #172033;
        color: white;
        padding: 35px 30px;
        border-radius: 18px;
        min-height: 570px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    }

    .college-name {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .department {
        font-size: 16px;
        color: #cbd5e1;
        margin-bottom: 28px;
    }

    .side-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .side-text {
        font-size: 15px;
        line-height: 1.8;
        color: #e2e8f0;
    }

    /* Right panel */
    .main-panel {
        background: white;
        padding: 45px 50px;
        border-radius: 18px;
        min-height: 570px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.10);
    }

    .project-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 25px;
    }

    .description-title {
        font-size: 22px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 10px;
    }

    .description {
        font-size: 16px;
        line-height: 1.7;
        color: #475569;
        margin-bottom: 35px;
    }

    .email-title {
        font-size: 18px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    </style>
    """, unsafe_allow_html=True)


    # =====================================================
    # TWO SIDE LAYOUT
    # =====================================================

    left_col, right_col = st.columns(
        [0.38, 0.62],
        gap="large"
    )


    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left_col:

        st.markdown("""
        <div class="info-panel">

            <div class="college-name">
                VSM College of Engineering
            </div>

            <div class="department">
                Autonomous
            </div>

            <div class="side-title">
                Department
            </div>

            <div class="side-text">
                Computer Science and Engineering (CSE)
            </div>

            <div class="side-title">
                Project Title
            </div>

            <div class="side-text">
                CCTV Weapon Detection System
            </div>

            <div class="side-title">
                Team Members
            </div>

            <div class="side-text">
                S. Nagasindhu – Team Member<br>
                S. Bhavyasri – Team Member<br>
                S. Manasa – Team Member<br>
                S. Anusha – Team Member
            </div>

            <div class="side-title">
                Project Guide
            </div>

            <div class="side-text">
                Mr. Abdul Aziz MD
            </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right_col:

        st.markdown("""
        <div class="main-panel">

            <div class="project-title">
                🚨 CCTV Weapon Detection System
            </div>

            <div class="description-title">
                Description
            </div>

            <div class="description">
                This project is an AI-powered CCTV weapon detection
                system designed to identify weapons from uploaded
                images and CCTV videos using a YOLO-based deep
                learning model. When a weapon is detected, the system
                provides a visual warning and generates an alert to
                support quick security response.
            </div>

        </div>
        """, unsafe_allow_html=True)


        st.markdown(
            '<div class="email-title">📧 Enter Email for Security Alerts</div>',
            unsafe_allow_html=True
        )

        email = st.text_input(
            "Email Address",
            placeholder="Enter your email address",
            label_visibility="collapsed"
        )


        # =================================================
        # NEXT BUTTON
        # =================================================

        if st.button(
            "NEXT  →",
            use_container_width=True,
            type="primary"
        ):

            # Validate email
            email_pattern = (
                r"^[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
            )

            if not email:

                st.error(
                    "⚠️ Please enter your email address."
                )

            elif not re.match(
                email_pattern,
                email
            ):

                st.error(
                    "⚠️ Please enter a valid email address."
                )

            else:

                # Save email
                st.session_state.alert_email = email

                # Open main project
                st.session_state.project_started = True

                st.rerun()


    # =====================================================
    # STOP MAIN PROJECT
    # =====================================================

    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=CONFIDENCE_THRESHOLD,
    step=0.05
)

required_frames = st.sidebar.slider(
    "Required Consecutive Frames",
    min_value=1,
    max_value=15,
    value=REQUIRED_CONSECUTIVE_FRAMES,
    step=1
)

st.sidebar.markdown("---")

st.sidebar.write("🤖 Model: YOLO")
st.sidebar.write("🎯 Detection: Weapon")
st.sidebar.write("📹 Input: CCTV Video")

# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("🖼️ Upload Image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

if uploaded_image is not None:

    st.success("✅ Image uploaded successfully!")

    # Read image
    from PIL import Image

    image = Image.open(uploaded_image)

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Detection button
    if st.button(
        "🔍 Detect Weapon in Image",
        use_container_width=True
    ):

        results = model(
            image,
            conf=confidence_threshold,
            verbose=False
        )

        weapon_detected = False
        detected_confidence = 0

        # ---------------------------------------------
        # CHECK DETECTIONS
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

                class_name = model.names[
                    class_id
                ]

                if (
                    class_name.lower() == "weapon"
                    and confidence >= confidence_threshold
                ):

                    weapon_detected = True

                    detected_confidence = confidence

        # ---------------------------------------------
        # DISPLAY RESULT
        # ---------------------------------------------

        if weapon_detected:

            st.error(
                f"🚨 WEAPON DETECTED! "
                f"Confidence: {detected_confidence:.2f}"
            )

        if os.path.exists("alert.mp3"):
            with open("alert.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(
                    audio_bytes,
                    format="audio/mp3"
                )

                st.warning("🔊 Click the Play button above to hear the alert sound.")

        else:

            st.success(
                "✅ NO WEAPON DETECTED"
            )

        # ---------------------------------------------
        # DISPLAY DETECTED IMAGE
        # ---------------------------------------------

        for result in results:

            annotated_image = result.plot()

            st.image(
                annotated_image,
                caption="Weapon Detection Result",
                use_container_width=True
            )

# =========================================================
# VIDEO UPLOAD
# =========================================================

st.subheader("🎥 Upload CCTV Video")

uploaded_video = st.file_uploader(
    "Choose a CCTV video",
    type=["mp4", "avi", "mov", "mkv"]
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
# PROCESS VIDEO
# =========================================================

if uploaded_video is not None:

    st.success("✅ Video uploaded successfully!")

    # -----------------------------------------------------
    # SAVE INPUT VIDEO TEMPORARILY
    # -----------------------------------------------------

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.read()
    )

    input_file.close()

    input_video_path = input_file.name

    # -----------------------------------------------------
    # OUTPUT FILE
    # -----------------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_video_path = output_file.name

    output_file.close()

    # -----------------------------------------------------
    # START BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Start Weapon Detection",
        use_container_width=True
    ):

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

        # -------------------------------------------------
        # VIDEO WRITER
        # -------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )

        # -------------------------------------------------
        # VARIABLES
        # -------------------------------------------------

        frame_number = 0

        consecutive_weapon_frames = 0

        confirmed_detections = 0

        last_box = None

        detection_events = []

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        progress_bar = st.progress(0)

        status_text = st.empty()

        video_placeholder = st.empty()

        # -------------------------------------------------
        # PROCESS FRAMES
        # -------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            current_weapon = None

            current_confidence = 0

            # =============================================
            # YOLO
            # =============================================

            results = model(
                frame,
                conf=confidence_threshold,
                verbose=False
            )

            # =============================================
            # FIND WEAPON
            # =============================================

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
                        class_name.lower()
                        == "weapon"
                        and confidence
                        >= confidence_threshold
                    ):

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        current_weapon = (
                            x1,
                            y1,
                            x2,
                            y2
                        )

                        current_confidence = (
                            confidence
                        )

                        break

            # =============================================
            # CONSECUTIVE DETECTION
            # =============================================

            if current_weapon is not None:

                if last_box is not None:

                    movement = distance(
                        last_box,
                        current_weapon
                    )

                    if movement < 100:

                        consecutive_weapon_frames += 1

                    else:

                        consecutive_weapon_frames = 1

                else:

                    consecutive_weapon_frames = 1

                last_box = current_weapon

            else:

                consecutive_weapon_frames = 0

                last_box = None

            # =============================================
            # CONFIRM
            # =============================================

            weapon_confirmed = (
                consecutive_weapon_frames
                >= required_frames
            )

            # =============================================
            # DRAW RESULT
            # =============================================

            if weapon_confirmed:

                x1, y1, x2, y2 = current_weapon

                confirmed_detections += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                label = (
                    f"WEAPON CONFIRMED "
                    f"{current_confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                time_sec = (
                    frame_number / fps
                )

                detection_events.append({
                    "time": round(time_sec, 2),
                    "confidence": round(
                        current_confidence,
                        2
                    )
                })

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

            # =============================================
            # WRITE FRAME
            # =============================================

            out.write(frame)

            # =============================================
            # PROGRESS
            # =============================================

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

        # -------------------------------------------------
        # RELEASE
        # -------------------------------------------------

        cap.release()

        out.release()

        progress_bar.progress(1.0)

        status_text.success(
            "✅ Video processing completed!"
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
        # DETECTION EVENTS
        # =================================================

        if detection_events:

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

        # =================================================
        # OUTPUT VIDEO
        # =================================================

        st.subheader(
            "🎥 Processed CCTV Video"
        )

        if os.path.exists(
            output_video_path
        ):

            with open(
                output_video_path,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="weapon_detection_result.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        # =================================================
        # CLEAN TEMP INPUT
        # =================================================

        try:

            os.remove(
                input_video_path
            )

        except:
            pass
