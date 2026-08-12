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