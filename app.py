import streamlit as st
import cv2
import os
import math
import tempfile
import smtplib

from email.message import EmailMessage
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Weapon Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "best.pt"
ALERT_SOUND = "alert.mp3"

CONFIDENCE_THRESHOLD = 0.60
REQUIRED_CONSECUTIVE_FRAMES = 5


# =========================================================
# SESSION STATE
# =========================================================

if "project_started" not in st.session_state:
    st.session_state.project_started = False


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* =====================================================
   FIRST PAGE
   ===================================================== */

.left-panel {
    background: #172033;
    color: white;
    padding: 35px 28px;
    border-radius: 18px;
    min-height: 620px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
}

.college-name {
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 5px;
}

.autonomous {
    color: #cbd5e1;
    font-size: 15px;
    margin-bottom: 30px;
}

.side-heading {
    font-size: 16px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 7px;
}

.side-text {
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.8;
}

.right-panel {
    background: white;
    padding: 42px;
    border-radius: 18px;
    min-height: 620px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.10);
}

.welcome-title {
    text-align: center;
    color: #172033;
    font-size: 35px;
    font-weight: 800;
    margin-bottom: 35px;
}

.description-title {
    color: #172033;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}

.description {
    color: #475569;
    font-size: 15px;
    line-height: 1.8;
}


/* =====================================================
   MAIN PROJECT
   ===================================================== */

.project-title {
    text-align: center;
    color: #172033;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.project-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    color: #172033;
    font-size: 28px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;
    margin-bottom: 20px;
}


/* =====================================================
   RESULT BOXES
   ===================================================== */

.danger-box {
    background-color: #fee2e2;
    border: 2px solid #ef4444;
    color: #b91c1c;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
}

.safe-box {
    background-color: #dcfce7;
    border: 2px solid #22c55e;
    color: #15803d;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    st.error("❌ best.pt not found!")

    st.info(
        "Please keep best.pt in the same folder as app.py."
    )

    st.stop()


@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# ALERT SOUND
# =========================================================

def play_alert_sound():

    if os.path.exists(ALERT_SOUND):

        with open(
            ALERT_SOUND,
            "rb"
        ) as audio_file:

            audio_bytes = audio_file.read()

        st.audio(
            audio_bytes,
            format="audio/mp3"
        )

    else:

        st.warning(
            "⚠️ alert.mp3 not found."
        )


# =========================================================
# EMAIL ALERT
# =========================================================

def send_email_alert(
    detection_type,
    confidence
):

    try:

        sender_email = st.secrets["ALERT_EMAIL"]

        sender_password = st.secrets[
            "ALERT_EMAIL_PASSWORD"
        ]

        receiver_email = st.secrets[
            "RECEIVER_EMAIL"
        ]

        message = EmailMessage()

        message["Subject"] = (
            "🚨 WEAPON GUARD AI - ALERT"
        )

        message["From"] = sender_email

        message["To"] = receiver_email

        message.set_content(
            f"""
WEAPON GUARD AI

🚨 WEAPON DETECTED

Detection Type:
{detection_type}

Confidence:
{confidence:.2f}

The Weapon Guard AI system detected
a possible weapon.

Please check the CCTV footage.

This is an automated security alert.
"""
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                sender_email,
                sender_password
            )

            smtp.send_message(message)

        return True

    except Exception:

        return False


# =========================================================
# BOX FUNCTIONS
# =========================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def distance(
    box1,
    box2
):

    c1 = box_center(box1)

    c2 = box_center(box2)

    return math.sqrt(
        (c1[0] - c2[0]) ** 2
        +
        (c1[1] - c2[1]) ** 2
    )


# =========================================================
# FIRST PAGE
# =========================================================

if not st.session_state.project_started:

    left_col, right_col = st.columns(
        [0.35, 0.65],
        gap="large"
    )


    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left_col:

        st.markdown("""
        <div class="left-panel">

            <div class="college-name">
                VSM College of Engineering
            </div>

            <div class="autonomous">
                Autonomous
            </div>

            <div class="side-heading">
                Department
            </div>

            <div class="side-text">
                Computer Science and Engineering
            </div>

            <div class="side-heading">
                Project Title
            </div>

            <div class="side-text">
                Weapon Guard AI
            </div>

            <div class="side-heading">
                Team
            </div>

            <div class="side-text">

                <b>Team Leader</b><br>
                S. Nagasindhu<br><br>

                <b>Team Members</b><br>
                S. Bhavyasri<br>
                S. Manasa<br>
                S. Anusha

            </div>

            <div class="side-heading">
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
        <div class="right-panel">

            <div class="welcome-title">
                🛡️ Weapon Guard AI
            </div>

            <div class="description-title">
                Problem Statement & Proposed Solution
            </div>

            <div class="description">

                Traditional CCTV monitoring depends heavily on
                continuous human observation, making it difficult
                to identify weapons in crowded environments and
                situations requiring constant attention. Manual
                monitoring can also delay the response to
                potentially dangerous situations. The proposed
                Weapon Guard AI system uses a YOLO-based deep
                learning model to automatically detect weapons
                from CCTV images and videos. When a weapon is
                detected, the system displays a bounding box and
                confidence score, provides a visual warning,
                plays an alert sound, and can send an email
                notification to a configured security address.
                This helps improve monitoring efficiency and
                supports faster security response.

            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button(
            "Next  →",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.project_started = True

            st.rerun()


    st.stop()


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="project-title">'
    '🛡️ Weapon Guard AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="project-subtitle">'
    'AI-powered weapon detection using YOLO'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

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

    st.write("🤖 Model: YOLO")
    st.write("🎯 Detection: Weapon")
    st.write("🖼️ Image")
    st.write("🎥 Video")


# =========================================================
# IMAGE DETECTION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🖼️ Image Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload an image to detect a weapon'
    '</div>',
    unsafe_allow_html=True
)


uploaded_image = st.file_uploader(
    "Choose CCTV Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="image_upload"
)


if uploaded_image is not None:

    image_input_col, image_output_col = st.columns(
        2,
        gap="large"
    )


    # =====================================================
    # INPUT IMAGE
    # =====================================================

    with image_input_col:

        st.subheader("📥 Input Image")

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )


    # =====================================================
    # OUTPUT IMAGE
    # =====================================================

    with image_output_col:

        st.subheader("📤 Output Image")

        detect_image = st.button(
            "🔍 Detect Weapon",
            use_container_width=True,
            key="detect_image"
        )

        if detect_image:

            results = model(
                image,
                conf=confidence_threshold,
                verbose=False
            )

            weapon_detected = False

            highest_confidence = 0.0

            output_image = image.copy()


            # =================================================
            # DRAW BOUNDING BOXES MANUALLY
            # =================================================

            output_array = cv2.cvtColor(
                output_image,
                cv2.COLOR_RGB2BGR
            )


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

                        weapon_detected = True

                        highest_confidence = max(
                            highest_confidence,
                            confidence
                        )


                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )


                        # -----------------------------------------
                        # RED BOUNDING BOX
                        # -----------------------------------------

                        cv2.rectangle(
                            output_array,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            4
                        )


                        # -----------------------------------------
                        # LABEL
                        # -----------------------------------------

                        label = (
                            f"WEAPON "
                            f"{confidence:.2f}"
                        )


                        cv2.putText(
                            output_array,
                            label,
                            (
                                x1,
                                max(
                                    y1 - 10,
                                    30
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )


            output_image = cv2.cvtColor(
                output_array,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # SHOW OUTPUT
            # =================================================

            st.image(
                output_image,
                use_container_width=True
            )


            # =================================================
            # DETECTION RESULT
            # =================================================

            if weapon_detected:

                st.markdown(
                    f"""
                    <div class="danger-box">
                        🚨 WEAPON DETECTED<br>
                        Confidence:
                        {highest_confidence:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                st.subheader(
                    "🔊 Security Alert"
                )

                play_alert_sound()


                # -------------------------------------------------
                # EMAIL
                # -------------------------------------------------

                email_sent = send_email_alert(
                    "Image",
                    highest_confidence
                )

                if email_sent:

                    st.success(
                        "📧 Email alert sent successfully."
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


# =========================================================
# VIDEO DETECTION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🎥 Video Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Input and processed output are displayed side-by-side'
    '</div>',
    unsafe_allow_html=True
)


uploaded_video = st.file_uploader(
    "Choose CCTV Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ],
    key="video_upload"
)


if uploaded_video is not None:

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
    # INPUT / OUTPUT COLUMNS
    # =====================================================

    input_video_col, output_video_col = st.columns(
        2,
        gap="large"
    )


    # =====================================================
    # INPUT VIDEO
    # =====================================================

    with input_video_col:

        st.subheader(
            "📥 Input Video"
        )

        st.video(
            input_video_path
        )


    # =====================================================
    # START DETECTION
    # =====================================================

    start_video = st.button(
        "🔍 Start Weapon Detection",
        use_container_width=True,
        key="start_video"
    )


    if start_video:

        # =================================================
        # OUTPUT FILE
        # =================================================

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_video_path = output_file.name

        output_file.close()


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


        # =================================================
        # VIDEO WRITER
        # =================================================

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
        # ========
