import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO


MODEL_PATH = Path("models/basma_yolo.pt")


@st.cache_resource
def load_yolo_model():
    return YOLO(str(MODEL_PATH))


def render_live_classroom():

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        '<div class="page-title">'
        'Classroom Video Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Upload a classroom video to analyze student '
        'activities and attendance.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # UPLOAD
    # =====================================================

    uploaded_file = st.file_uploader(
        "📹 Upload Classroom Video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a classroom recording."
    )

    # =====================================================
    # NOTHING UPLOADED
    # =====================================================

    if uploaded_file is None:

        st.info(
            "Please upload a classroom video to begin."
        )

        return

    # =====================================================
    # SHOW ORIGINAL VIDEO
    # =====================================================

    st.markdown("### 🎥 Video Preview")

    video_bytes = uploaded_file.getvalue()

    st.video(video_bytes)

    st.success(
        f"Video uploaded: {uploaded_file.name}"
    )

    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    ):

        analyze_video(
            uploaded_file
        )


# =========================================================
# VIDEO ANALYSIS
# =========================================================

def analyze_video(uploaded_file):

    st.markdown("### 🤖 AI Analysis")

    progress = st.progress(0)

    status = st.empty()

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

    suffix = Path(
        uploaded_file.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        input_path = Path(
            temp_file.name
        )

    # =====================================================
    # OUTPUT FILE
    # =====================================================

    output_path = Path(
        tempfile.mktemp(
            suffix=".mp4"
        )
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    status.write(
        "Loading BASMA AI model..."
    )

    try:

        model = load_yolo_model()

    except Exception as e:

        st.error(
            f"Could not load YOLO model: {e}"
        )

        return

    # =====================================================
    # OPEN VIDEO
    # =====================================================

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        st.error(
            "Could not open the uploaded video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

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

    # =====================================================
    # VIDEO WRITER
    # =====================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        st.error(
            "Could not create the output video."
        )

        return

    # =====================================================
    # ACTIVITY COUNTS
    # =====================================================

    activity_counts = {}

    frame_number = 0

    # =====================================================
    # PROCESS VIDEO
    # =====================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # -------------------------------------------------
        # YOLO
        # -------------------------------------------------

        try:

            results = model.predict(
                frame,
                conf=0.40,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]

            # -------------------------------------------------
            # Count activities
            # -------------------------------------------------

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    activity = model.names[
                        class_id
                    ]

                    activity_counts[
                        activity
                    ] = (
                        activity_counts.get(
                            activity,
                            0
                        )
                        + 1
                    )

            # -------------------------------------------------
            # Draw boxes
            # -------------------------------------------------

            annotated_frame = result.plot()

        except Exception:

            annotated_frame = frame

        # -------------------------------------------------
        # Write frame
        # -------------------------------------------------

        writer.write(
            annotated_frame
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if total_frames > 0:

            percent = (
                frame_number
                / total_frames
            )

            progress.progress(
                min(percent, 1.0)
            )

            status.write(
                f"Analyzing video... "
                f"{int(percent * 100)}%"
            )

    # =====================================================
    # CLOSE
    # =====================================================

    cap.release()

    writer.release()

    progress.progress(1.0)

    status.success(
        "Video analysis completed!"
    )

    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown(
        "### 🎬 AI Annotated Video"
    )

    if output_path.exists():

        with open(
            output_path,
            "rb"
        ) as file:

            video_bytes = file.read()

        st.video(
            video_bytes
        )

        st.download_button(
            "⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="basma_annotated_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # =====================================================
    # ACTIVITY SUMMARY
    # =====================================================

    st.markdown(
        "### 📊 Activity Summary"
    )

    if activity_counts:

        # Sort by most detected
        sorted_activities = sorted(
            activity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for activity, count in sorted_activities:

            st.write(
                f"**{activity}** — {count} detections"
            )

    else:

        st.info(
            "No classroom activities were detected."
        )
