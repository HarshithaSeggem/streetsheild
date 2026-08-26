import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import time
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StreetShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIMPLE PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: white;
}

p, label {
    color: #cbd5e1;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #263449;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: white;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
}

[data-testid="stFileUploader"] {
    background-color: #111827;
    border: 1px dashed #3b82f6;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/StreetShield_final.keras"
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_detector():

    detector = cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    return detector


# ============================================================
# LOAD
# ============================================================

try:
    model = load_model()
    detector = load_detector()
    system_ready = True

except Exception as e:

    system_ready = False
    st.error("Unable to load StreetShield models.")
    st.code(str(e))
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ StreetShield")

    st.caption("AI MEDIA FORENSICS PLATFORM")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔎 Analyze Media",
            "📊 Analysis History",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.subheader("System Status")

    if system_ready:
        st.success("Face Detector Online")
        st.success("AI Classifier Online")
        st.success("Analysis Engine Ready")
    else:
        st.error("System Offline")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🛡️ StreetShield")

    st.subheader("Verify Before You Trust.")

    st.write(
        "An AI-powered media verification platform that "
        "detects human faces and analyzes them for potential "
        "AI-generated or manipulated content."
    )

    st.divider()

    # Statistics

    total = len(st.session_state.history)

    real_count = sum(
        1 for x in st.session_state.history
        if x["result"] == "REAL"
    )

    fake_count = sum(
        1 for x in st.session_state.history
        if x["result"] == "AI-GENERATED"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "System",
        "ONLINE"
    )

    c2.metric(
        "Model",
        "EfficientNetB0"
    )

    c3.metric(
        "Analyses",
        total
    )

    c4.metric(
        "AI Detected",
        fake_count
    )

    st.divider()

    st.header("How StreetShield Works")

    a, b, c, d, e = st.columns(5)

    with a:
        st.markdown("### 01")
        st.write("📤 Upload")
        st.caption("Upload a face image.")

    with b:
        st.markdown("### 02")
        st.write("👁️ Detect")
        st.caption("YuNet detects faces.")

    with c:
        st.markdown("### 03")
        st.write("✂️ Crop")
        st.caption("Face region is extracted.")

    with d:
        st.markdown("### 04")
        st.write("🧠 Analyze")
        st.caption("EfficientNetB0 analyzes the face.")

    with e:
        st.markdown("### 05")
        st.write("🛡️ Verify")
        st.caption("Prediction and confidence are generated.")

    st.divider()

    st.header("Key Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("👤 Face Detection")

        st.write(
            "Detects human faces before classification. "
            "Images without detectable faces are rejected."
        )

    with c2:

        st.subheader("🧠 AI Classification")

        st.write(
            "EfficientNetB0 analyzes the detected face "
            "and predicts REAL or AI-GENERATED."
        )

    with c3:

        st.subheader("📊 Explainable Results")

        st.write(
            "Displays confidence, face count, image "
            "dimensions and processing time."
        )

    st.divider()

    st.header("Designed For")

    u1, u2, u3, u4 = st.columns(4)

    with u1:
        st.subheader("📰 Journalism")
        st.caption(
            "Support media verification before publication."
        )

    with u2:
        st.subheader("📱 Social Media")
        st.caption(
            "Assess suspicious AI-generated profile images."
        )

    with u3:
        st.subheader("🔐 Digital Safety")
        st.caption(
            "Help users evaluate suspicious face media."
        )

    with u4:
        st.subheader("🎓 Research")
        st.caption(
            "Academic research and deepfake detection."
        )

# ============================================================
# ANALYZE MEDIA
# ============================================================

elif page == "🔎 Analyze Media":

    st.title("🔎 Analyze Media")

    st.write(
        "Upload an image containing a human face."
    )

    st.info(
        "StreetShield currently analyzes human-face images only."
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        height, width = image_np.shape[:2]

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        detector.setInputSize(
            (width, height)
        )

        start = time.time()

        _, faces = detector.detect(
            image_np
        )

        detection_time = time.time() - start

        # ----------------------------------------------------
        # IMAGE INFORMATION
        # ----------------------------------------------------

        st.divider()

        st.header("Media Information")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Width",
            f"{width}px"
        )

        c2.metric(
            "Height",
            f"{height}px"
        )

        if faces is not None:
            face_count = len(faces)
        else:
            face_count = 0

        c3.metric(
            "Faces Detected",
            face_count
        )

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with right:

            st.subheader("Detected Face")

            if face_count > 0:

                # Largest face

                face = max(
                    faces,
                    key=lambda f: f[2] * f[3]
                )

                x, y, w, h = face[:4].astype(int)

                x = max(0, x)
                y = max(0, y)

                x2 = min(
                    width,
                    x + w
                )

                y2 = min(
                    height,
                    y + h
                )

                face_crop = image_np[
                    y:y2,
                    x:x2
                ]

                st.image(
                    face_crop,
                    caption=f"{face_count} face(s) detected",
                    use_container_width=True
                )

            else:

                face_crop = None

                st.error(
                    "No human face detected."
                )

                st.stop()

        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        st.divider()

        st.header("AI Authenticity Analysis")

        analyze = st.button(
            "🔍 ANALYZE IMAGE"
        )

        if analyze:

            with st.spinner(
                "Analyzing image..."
            ):

                start_prediction = time.time()

                resized = cv2.resize(
                    face_crop,
                    (224, 224)
                )

                resized = resized.astype(
                    np.float32
                )

                input_image = np.expand_dims(
                    resized,
                    axis=0
                )

                prediction = model.predict(
                    input_image,
                    verbose=0
                )[0][0]

                prediction_time = (
                    time.time()
                    - start_prediction
                )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction >= 0.5:

                result = "REAL"

                confidence = prediction * 100

                st.success(
                    f"🟢 REAL — {confidence:.2f}% confidence"
                )

            else:

                result = "AI-GENERATED"

                confidence = (
                    1 - prediction
                ) * 100

                st.error(
                    f"🔴 AI-GENERATED — "
                    f"{confidence:.2f}% confidence"
                )

            st.progress(
                int(confidence)
            )

            # ------------------------------------------------
            # RESULT METRICS
            # ------------------------------------------------

            st.subheader("Analysis Details")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Prediction",
                result
            )

            c2.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            c3.metric(
                "Faces",
                face_count
            )

            c4.metric(
                "Processing",
                f"{prediction_time + detection_time:.2f}s"
            )

            # ------------------------------------------------
            # TECHNICAL DETAILS
            # ------------------------------------------------

            st.subheader("Technical Details")

            t1, t2 = st.columns(2)

            with t1:

                st.write(
                    "**Face Detector:** YuNet"
                )

                st.write(
                    "**Classifier:** EfficientNetB0"
                )

                st.write(
                    "**Input Size:** 224 × 224"
                )

            with t2:

                st.write(
                    "**Image Size:** "
                    f"{width} × {height}"
                )

                st.write(
                    "**Faces Detected:** "
                    f"{face_count}"
                )

                st.write(
                    "**Analysis Status:** Completed"
                )

            st.warning(
                "The prediction is probabilistic. "
                "It should be treated as supporting evidence "
                "rather than definitive proof of authenticity."
            )

            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            st.session_state.history.insert(
                0,
                {
                    "result": result,
                    "confidence": confidence,
                    "faces": face_count,
                    "dimensions": f"{width} × {height}",
                    "time": prediction_time + detection_time
                }
            )

# ============================================================
# HISTORY
# ============================================================

elif page == "📊 Analysis History":

    st.title("📊 Analysis History")

    history = st.session_state.history

    if not history:

        st.info(
            "No analysis has been performed yet."
        )

    else:

        total = len(history)

        real_count = sum(
            x["result"] == "REAL"
            for x in history
        )

        fake_count = sum(
            x["result"] == "AI-GENERATED"
            for x in history
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Analyses",
            total
        )

        c2.metric(
            "REAL",
            real_count
        )

        c3.metric(
            "AI-GENERATED",
            fake_count
        )

        st.divider()

        for index, item in enumerate(history):

            if item["result"] == "REAL":
                icon = "🟢"
            else:
                icon = "🔴"

            with st.container(border=True):

                c1, c2, c3, c4 = st.columns(4)

                c1.write(
                    f"{icon} **{item['result']}**"
                )

                c2.write(
                    f"Confidence: **{item['confidence']:.2f}%**"
                )

                c3.write(
                    f"Faces: **{item['faces']}**"
                )

                c4.write(
                    f"Time: **{item['time']:.2f}s**"
                )

        st.divider()

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()

# ============================================================
# ABOUT
# ============================================================

else:

    st.title("ℹ️ About StreetShield")

    st.subheader(
        "AI-Powered Deepfake Detection and Media Verification"
    )

    st.write(
        "StreetShield is a computer-vision and deep-learning "
        "based platform designed to analyze human-face images "
        "and classify them as REAL or potentially "
        "AI-GENERATED."
    )

    st.divider()

    st.header("System Architecture")

    st.code(
        """
Image Upload
     ↓
YuNet Face Detection
     ↓
Face Cropping
     ↓
224 × 224 Preprocessing
     ↓
EfficientNetB0
     ↓
REAL / AI-GENERATED
     ↓
Confidence & Analysis Report
        """,
        language="text"
    )

    st.header("Technology Stack")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Language",
        "Python"
    )

    c2.metric(
        "Computer Vision",
        "OpenCV"
    )

    c3.metric(
        "Deep Learning",
        "TensorFlow"
    )

    c4.metric(
        "Model",
        "EfficientNetB0"
    )

    st.divider()

    st.header("Project Use Cases")

    st.write("📰 **Journalism** — Media verification")

    st.write("📱 **Social Media** — Suspicious image assessment")

    st.write("🔐 **Digital Safety** — Face-media verification")

    st.write("🎓 **Research** — Deepfake detection research")

    st.divider()

    st.header("Important Limitation")

    st.warning(
        "StreetShield provides a machine-learning based "
        "probabilistic assessment. A prediction should not "
        "be considered absolute proof that an image is "
        "real or fake."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ StreetShield | AI Media Forensics Platform | "
    "Academic Project"
)