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
