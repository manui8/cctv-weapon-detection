import streamlit as st
import cv2
import os
import math
import tempfile
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
ALERT_SOUND = "alert.mp3"

CONFIDENCE_THRESHOLD = 0.60
REQUIRED_CONSECUTIVE_FRAMES = 5


# =========================================================
# TEAM DETAILS
# =========================================================

TEAM_LEAD = "S. Nagasindhu"

TEAM_MEMBERS = [
    "S. Bhavyasri",
    "S. Manasa",
    "S. Anusha"
]

COLLEGE_NAME = "VSM College of Engineering"
GUIDE_NAME = "Mr. Abdul Aziz MD"


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ================================
       GENERAL
       ================================ */

    .stApp {
        background-color: #0e1117;
    }

    /* Remove top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* ================================
       WELCOME PAGE
       ================================ */

    .welcome-left {
        padding: 35px 30px;
        min-height: 600px;
    }

    .welcome-right {
        padding: 50px 55px;
        min-height: 600px;
    }

    .aicw-heading {
        font-size: 26px;
        font-weight: 700;
        line-height: 1.3;
        color: #ffffff;
        margin-bottom: 25px;
    }

    .college-heading {
        font-size: 22px;
        font-weight: 600;
        color: #4da6ff;
        margin-bottom: 35px;
    }

    .team-heading {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 22px;
    }

    .team-lead {
        font-size: 18px;
        font-weight: 700;
        color: #50fa7b;
        margin-bottom: 15px;
    }

    .team-member {
        font-size: 17px;
        color: #dddddd;
        margin-bottom: 12px;
    }

    .guide-heading {
        font-size: 17px;
        color: #aaaaaa;
        margin-top: 35px;
        margin-bottom: 6px;
    }

    .guide-name {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
    }

    .weapon-title {
        font-size: 46px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        margin-bottom: 12px;
    }

    .description-heading {
        font-size: 25px;
        font-weight: 700;
        color: #4da6ff;
        margin-top: 35px;
        margin-bottom: 14px;
    }

    .description-text {
        font-size: 18px;
        line-height: 1.75;
        color: #dddddd;
        text-align: justify;
        max-width: 800px;
    }


    /* ================================
       DETECTION PAGE
       ================================ */

    .detection-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .detection-subtitle {
        font-size: 18px;
        text-align: center;
        color: #aaaaaa;
        margin-bottom: 30px;
    }

    .result-safe {
        background-color: #123d24;
        color: #50fa7b;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
    }

    .result-danger {
        background-color: #4a1111;
        color: #ff5555;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }

    /* Keep video/image containers neat */
    .video-box {
        width: 100%;
        aspect-ratio: 4 / 3;
        overflow: hidden;
        border-radius: 10px;
    }


    /* ================================
       BUTTONS
       ================================ */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
    }

    </style>
    """,
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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# =========================================================
# ALERT SOUND
# =========================================================

def play_alert():

    if os.path.exists(ALERT_SOUND):

        with open(ALERT_SOUND, "rb") as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/mp3"
            )


# =========================================================
# HOME / WELCOME PAGE
# =========================================================

def home_page():

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown(
            """
            <div class="aicw-heading">
                Artificial Intelligence Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            '<div class="college-heading">VSM College of Engineering</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="team-heading">Team Members</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="team-lead">{TEAM_LEAD} — Team Lead</div>',
            unsafe_allow_html=True
        )

        for member in TEAM_MEMBERS:

            st.markdown(
                f'<div class="team-member">{member} — Team Member</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="guide-heading">Project Guide</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="guide-name">{GUIDE_NAME}</div>',
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # MAIN PAGE
    # -----------------------------------------------------

    left_col, right_col = st.columns(
        [0.9, 1.5],
        gap="large"
    )


    # -----------------------------------------------------
    # LEFT SIDE
    # -----------------------------------------------------

    with left_col:

        st.markdown(
            '<div class="welcome-left">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="aicw-heading">
                Artificial Intelligence Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="college-heading">
                {COLLEGE_NAME}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="team-heading">
                Team Members
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="team-lead">
                {TEAM_LEAD} — Team Lead
            </div>
            """,
            unsafe_allow_html=True
        )

        for member in TEAM_MEMBERS:

            st.markdown(
                f"""
                <div class="team-member">
                    {member} — Team Member
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="guide-heading">
                Project Guide
            </div>

            <div class="guide-name">
                {GUIDE_NAME}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # RIGHT SIDE
    # -----------------------------------------------------

    with right_col:

        st.markdown(
            '<div class="welcome-right">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="weapon-title">
                WeaponGuard AI
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="description-heading">
                Description
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
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
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br><br>", unsafe_allow_html=True)

        if st.button(
            "NEXT ➜",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.page = "detection"

            st.rerun()

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# IMAGE DETECTION
# =========================================================

def image_detection(model, confidence_threshold):

    st.markdown(
        '<div class="section-title">📷 Image Detection</div>',
        unsafe_allow_html=True
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


    st.success("Image uploaded successfully.")


    if st.button(
        "🔍 Detect Weapon",
        use_container_width=True,
        key="detect_image_button"
    ):

        image = Image.open(uploaded_image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_array = np.array(image)

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )


        # -----------------------------------------------
        # YOLO DETECTION
        # -----------------------------------------------

        results = model(
            image_bgr,
            conf=0.10,
            verbose=False
        )


        output_image = image_bgr.copy()

        weapon_count = 0


        # -----------------------------------------------
        # DRAW DETECTIONS
        # -----------------------------------------------

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


        # -----------------------------------------------
        # INPUT / OUTPUT SIDE BY SIDE
        # -----------------------------------------------

        st.markdown("---")

        input_col, output_col = st.columns(
            2,
            gap="medium"
        )


        with input_col:

            st.markdown(
                '<div class="section-title">📥 Input Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_array,
                use_container_width=True
            )


        with output_col:

            st.markdown(
                '<div class="section-title">📤 Output Image</div>',
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


        # -----------------------------------------------
        # RESULT
        # -----------------------------------------------

        if weapon_count > 0:

            st.markdown(
                f"""
                <div class="result-danger">
                    🚨 {weapon_count} WEAPON(S) DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

            play_alert()

        else:

            st.markdown(
                """
                <div class="result-safe">
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

    st.markdown(
        '<div class="section-title">🎥 Video Detection</div>',
        unsafe_allow_html=True
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


    # -----------------------------------------------------
    # READ VIDEO
    # -----------------------------------------------------

    video_bytes = uploaded_video.getvalue()


    # -----------------------------------------------------
    # SAVE INPUT TEMPORARILY
    # -----------------------------------------------------

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(video_bytes)
    input_file.close()

    input_video_path = input_file.name


    # -----------------------------------------------------
    # SHOW ORIGINAL INPUT VIDEO
    # -----------------------------------------------------

    st.markdown("---")

    input_col, output_col = st.columns(
        2,
        gap="medium"
    )


    with input_col:

        st.markdown(
            '<div class="section-title">📥 Input Video</div>',
            unsafe_allow_html=True
        )

        # Browser can directly play uploaded video
        st.video(
            video_bytes
        )


    # -----------------------------------------------------
    # START DETECTION
    # -----------------------------------------------------

    with output_col:

        st.markdown(
            '<div class="section-title">📤 Output Video</div>',
            unsafe_allow_html=True
        )

        output_placeholder = st.empty()

        output_placeholder.info(
            "Click **Start Weapon Detection** "
            "to generate the processed video."
        )


    st.markdown("")


    start_detection = st.button(
        "🔍 Start Weapon Detection",
        use_container_width=True,
        type="primary",
        key="start_video_detection"
    )


    if not start_detection:
        return


    # -----------------------------------------------------
    # OUTPUT TEMP FILE
    # -----------------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_video_path = output_file.name

    output_file.close()


    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        input_video_path
    )


    if not cap.isOpened():

        st.error(
            "❌ Could not open the uploaded video."
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
    # VIDEO WRITER
    # -----------------------------------------------------

    # Try mp4v first
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )


    if not out.isOpened():

        cap.release()

        st.error(
            "❌ Could not create output video."
        )

        return


    # -----------------------------------------------------
    # TRACKING VARIABLES
    # -----------------------------------------------------

    frame_number = 0

    consecutive_weapon_frames = {}

    last_boxes = {}

    confirmed_detections = 0

    detection_events = []

    weapon_detected_anywhere = False


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


        current_weapons = []


        # -----------------------------------------------
        # YOLO
        # -----------------------------------------------

        results = model(
            frame,
            conf=confidence_threshold,
            verbose=False
        )


        # -----------------------------------------------
        # FIND WEAPONS
        # -----------------------------------------------

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


        # -----------------------------------------------
        # TRACK WEAPONS
        # -----------------------------------------------

        confirmed_weapon_boxes = []

        new_last_boxes = {}

        used_previous_ids = set()


        for weapon in current_weapons:

            weapon_box = weapon["box"]

            weapon_confidence = weapon[
                "confidence"
            ]


            best_match_id = None

            best_match_distance = 100


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


            # -------------------------------------------
            # EXISTING WEAPON
            # -------------------------------------------

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
                            "confidence": weapon_confidence,
                            "id": best_match_id
                        }
                    )


            # -------------------------------------------
            # NEW WEAPON
            # -------------------------------------------

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


        # -----------------------------------------------
        # DRAW OUTPUT
        # -----------------------------------------------

        if confirmed_weapon_boxes:

            weapon_detected_anywhere = True

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
                        max(y1 - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


                detection_events.append(
                    {
                        "time": round(
                            frame_number / fps,
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


        # -----------------------------------------------
        # WRITE OUTPUT FRAME
        # -----------------------------------------------

        out.write(frame)


        # -----------------------------------------------
        # PROGRESS
        # -----------------------------------------------

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


    # -----------------------------------------------------
    # RELEASE
    # -----------------------------------------------------

    cap.release()
    out.release()


    progress_bar.progress(1.0)

    status_text.success(
        "✅ Video processing completed!"
    )


    # -----------------------------------------------------
    # READ OUTPUT VIDEO
    # -----------------------------------------------------

    if not os.path.exists(
        output_video_path
    ):

        st.error(
            "❌ Output video was not created."
        )

        return


    with open(
        output_video_path,
        "rb"
    ) as output_file:

        processed_video_bytes = (
            output_file.read()
        )


    # -----------------------------------------------------
    # DISPLAY OUTPUT VIDEO
    # -----------------------------------------------------

    output_placeholder.empty()


    with output_col:

        st.markdown(
            '<div class="section-title">📤 Output Video</div>',
            unsafe_allow_html=True
        )

        # Browser-playable video
        st.video(
            processed_video_bytes
        )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown("---")

    if weapon_detected_anywhere:

        st.markdown(
            """
            <div class="result-danger">
                🚨 WEAPON DETECTED IN VIDEO
            </div>
            """,
            unsafe_allow_html=True
        )

        # Siren plays only after video processing
        play_alert()


    else:

        st.markdown(
            """
            <div class="result-safe">
                ✅ NO WEAPON DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # DETECTION DETAILS
    # -----------------------------------------------------

    if detection_events:

        st.markdown("---")

        st.subheader(
            "🚨 Detection Events"
        )


        # Show limited events to keep UI clean
        displayed_events = detection_events[:30]


        for event in displayed_events:

            st.warning(
                f"⏱️ Time: "
                f"{event['time']} sec  |  "
                f"Confidence: "
                f"{event['confidence']}"
            )


    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.download_button(
        label="⬇️ Download Processed Video",
        data=processed_video_bytes,
        file_name="weapon_detection_result.mp4",
        mime="video/mp4",
        use_container_width=True
    )


    # -----------------------------------------------------
    # CLEAN TEMP INPUT
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown(
            """
            <div class="aicw-heading">
                Artificial Intelligence Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")


        # BACK BUTTON

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
        st.write("📷 **Input:** Image / CCTV Video")


    # -----------------------------------------------------
    # CHECK MODEL
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
    # LOAD MODEL
    # -----------------------------------------------------

    model = load_model()


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------

    image_tab, video_tab = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection"
        ]
    )


    # -----------------------------------------------------
    # IMAGE TAB
    # -----------------------------------------------------

    with image_tab:

        image_detection(
            model,
            confidence_threshold
        )


    # -----------------------------------------------------
    # VIDEO TAB
    # -----------------------------------------------------

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
