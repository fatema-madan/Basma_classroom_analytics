import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

CONFIDENCE = 0.40


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_my_model():
    return YOLO(MODEL_PATH)


# =========================================================
# LIVE CAMERA
# =========================================================

def render_live_classroom():

    st.markdown("### 📸 Live Camera Detection")

    st.markdown(
        "Use the camera to capture a classroom image "
        "and BASMA will detect student activities."
    )


    # =====================================================
    # CAMERA
    # =====================================================

    cam_file = st.camera_input(
        "Take a photo"
    )


    # =====================================================
    # CHECK CAMERA INPUT
    # =====================================================

    if cam_file is not None:

        # -----------------------------------------------
        # Read camera image
        # -----------------------------------------------

        image = Image.open(
            cam_file
        ).convert("RGB")


        # =================================================
        # AI DETECTION
        # =================================================

        with st.spinner(
            "AI is analyzing..."
        ):

            model = load_my_model()

            result = model.predict(
                image,
                conf=CONFIDENCE,
                verbose=False
            )[0]


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.markdown(
            "### 🤖 AI Detection Result"
        )

        annotated_image = result.plot()

        st.image(
            annotated_image,
            caption="BASMA Activity Detection",
            use_container_width=True
        )


        # =================================================
        # DETECTED ACTIVITIES
        # =================================================

        detected_activities = []


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                activity = model.names[
                    class_id
                ]

                detected_activities.append(
                    {
                        "Activity": activity,
                        "Confidence": f"{confidence:.2%}"
                    }
                )


        # =================================================
        # SHOW RESULTS
        # =================================================

        if detected_activities:

            st.warning(
                "⚠️ Classroom activities detected!"
            )


            st.dataframe(
                detected_activities,
                use_container_width=True,
                hide_index=True
            )


            # ---------------------------------------------
            # Activity summary
            # ---------------------------------------------

            activity_names = sorted(
                set(
                    item["Activity"]
                    for item in detected_activities
                )
            )


            st.markdown(
                "### 📊 Detected Activities"
            )


            for activity in activity_names:

                st.write(
                    f"• {activity}"
                )


        else:

            st.success(
                "✅ No classroom activity detected."
            )


    else:

        st.info(
            "📷 Click **Take a photo** to start detection."
        )

