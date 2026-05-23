import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Titanic Survival AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).parent

model_path = BASE_DIR / "titanic_model.keras"
scaler_path = BASE_DIR / "scaler.pkl"

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
    <style>

    /* =========================================================
    GLOBAL
    ========================================================= */

    .stApp {
        background:
            radial-gradient(circle at top left, #f8fafc 0%, transparent 35%),
            radial-gradient(circle at bottom right, #dbeafe 0%, transparent 30%),
            linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #e0f2fe 100%);

        color: #111827;
    }

    header, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0rem;
        max-width: 1450px;
    }

    /* =========================================================
    TYPOGRAPHY
    ========================================================= */

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        color: #0f172a;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        display: block;
        text-align: center;
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
    }

    label, p {
        color: #111827 !important;
    }

    /* =========================================================
    HEADER
    ========================================================= */

    .title-box {
        background: rgba(255,255,255,0.55);

        border-radius: 30px;

        padding: 1.8rem;

        border: 1px solid rgba(255,255,255,0.60);

        backdrop-filter: blur(14px);

        box-shadow:
            0 10px 32px rgba(15,23,42,0.08),
            inset 0 1px 0 rgba(255,255,255,0.45);

        border-bottom: 2px solid rgba(71,85,105,0.18);

        margin-bottom: 1.5rem;
    }

    /* =========================================================
    MAIN PANELS
    ========================================================= */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.68);

        border: 1.5px solid rgba(148,163,184,0.18);

        border-radius: 28px;

        padding: 1.4rem;

        backdrop-filter: blur(18px);

        box-shadow:
            0 10px 30px rgba(15,23,42,0.10),
            inset 0 1px 0 rgba(255,255,255,0.45);

        transition: all 0.25s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);

        box-shadow:
            0 16px 40px rgba(15,23,42,0.14),
            inset 0 1px 0 rgba(255,255,255,0.55);
    }

    /* =========================================================
    COLUMN DIVIDER
    ========================================================= */

    div[data-testid="stHorizontalBlock"] > div:nth-child(1) {
        border-right: 1.5px solid rgba(100,116,139,0.20);
        padding-right: 2rem;
    }

    div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        padding-left: 2rem;
    }

    /* =========================================================
    SECTION HEADINGS
    ========================================================= */

    h3 {
        padding-bottom: 0.7rem;
        border-bottom: 1px solid rgba(148,163,184,0.20);
        margin-bottom: 1.3rem;
    }

    /* =========================================================
    SELECTBOX
    ========================================================= */

    .stSelectbox * {
        cursor: pointer !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.75) !important;

        border-radius: 16px !important;

        border: 1px solid rgba(255,255,255,0.45) !important;

        backdrop-filter: blur(10px) !important;

        min-height: 3rem;
    }

    /* =========================================================
    NUMBER INPUT
    ========================================================= */

    [data-testid="stNumberInput"] button {
        display: none !important;
    }

    div[data-baseweb="input"] {
        background: transparent !important;
    }

    div[data-baseweb="input"] > div {
        background: rgba(255,255,255,0.75) !important;

        border-radius: 16px !important;

        border: 1px solid rgba(255,255,255,0.45) !important;

        backdrop-filter: blur(10px) !important;

        min-height: 3rem;

        box-shadow: none !important;
    }

    div[data-baseweb="input"] input {
        background: transparent !important;
        border: none !important;
        color: #111827 !important;
        text-align: left;
        font-weight: 500;
    }

    /* =========================================================
    SLIDER
    ========================================================= */

    .stSlider label {
        font-weight: 600;
    }

    /* =========================================================
    BUTTON
    ========================================================= */

    .stButton > button {
        width: 100%;
        height: 3.4rem;

        border: none;
        border-radius: 18px;

        font-size: 1rem;
        font-weight: 700;

        background: linear-gradient(135deg, #2563eb, #1d4ed8);

        color: white;

        transition: all 0.25s ease;

        box-shadow: 0 8px 22px rgba(37,99,235,0.25);

        margin-top: 0.8rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow: 0 12px 28px rgba(37,99,235,0.35);
    }

    /* =========================================================
    METRICS
    ========================================================= */

    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.55);

        border: 1px solid rgba(255,255,255,0.55);

        border-radius: 20px;

        padding: 1rem;

        backdrop-filter: blur(10px);

        box-shadow: 0 4px 16px rgba(15,23,42,0.05);
    }

    /* =========================================================
    PREDICTION BOX
    ========================================================= */

    .prediction-box {
        padding: 24px;

        border-radius: 24px;

        text-align: center;

        font-size: 1.4rem;
        font-weight: 700;

        margin-top: 1.5rem;

        color: white;

        box-shadow: 0 10px 28px rgba(15,23,42,0.15);
    }

    .survive {
        background: linear-gradient(135deg, #16a34a, #22c55e);
    }

    .die {
        background: linear-gradient(135deg, #dc2626, #ef4444);
    }

    </style>
    """, unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="title-box">
        <h1 class="main-title">🚢 Titanic Survival AI</h1>
        <div class="subtitle">
            Deep Learning Powered Survival Prediction System
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LAYOUT
# =========================================================

left, right = st.columns([0.95, 1.05], gap="medium")

# =========================================================
# LEFT PANEL
# =========================================================

with left:

    with st.container(border=True):

        st.subheader("🧾 Passenger Information")

        col1, col2 = st.columns(2)

        # ---------------- LEFT INPUTS ---------------- #

        with col1:

            pclass = st.selectbox(
                "Passenger Class",
                [1, 2, 3]
            )

            sibsp = st.slider(
                "Siblings / Spouses",
                0,
                8,
                0
            )

            fare = st.number_input(
                "Ticket Fare",
                min_value=1,
                max_value=600,
                value=50,
                step=1
            )

        # ---------------- RIGHT INPUTS ---------------- #

        with col2:

            sex = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            parch = st.slider(
                "Parents / Children",
                0,
                6,
                0
            )

            embarked = st.selectbox(
                "Embarkation",
                ["Cherbourg", "Queenstown", "Southampton"]
            )

        predict_btn = st.button(
            "🔮 Predict Survival",
            use_container_width=True
        )

# =========================================================
# RIGHT PANEL
# =========================================================

with right:

    with st.container(border=True):

        st.subheader("📊 Model Insights")

        m1, m2 = st.columns(2)

        with m1:
            st.metric("Accuracy", "86%")

        with m2:
            st.metric("Framework", "TensorFlow")

        st.markdown("""
        ### 🧠 About

        Deep neural network trained on passenger demographics,
        fare data, and embarkation details to estimate survival probability.
        """)

        # =========================================================
        # PREDICTION
        # =========================================================

        if predict_btn:

            # ---------------- ENCODING ---------------- #

            sex = 1 if sex == "Male" else 0

            embarked_c = 1 if embarked == "Cherbourg" else 0
            embarked_q = 1 if embarked == "Queenstown" else 0
            embarked_s = 1 if embarked == "Southampton" else 0

            # ---------------- INPUT DATAFRAME ---------------- #

            input_df = pd.DataFrame({
                "Pclass": [pclass],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare]
            })

            # ---------------- SCALING ---------------- #

            scaled_array = scaler.transform(input_df)

            scaled_df = pd.DataFrame(
                scaled_array,
                columns=["Pclass", "SibSp", "Parch", "Fare"]
            )

            # ---------------- ENCODED FEATURES ---------------- #

            scaled_df["Sex"] = sex
            scaled_df["Embarked_Chebourg"] = embarked_c
            scaled_df["Embarked_Queenstown"] = embarked_q
            scaled_df["Embarked_Southhampton"] = embarked_s

            # ---------------- COLUMN ORDER ---------------- #

            scaled_df = scaled_df[
                [
                    "Pclass",
                    "Sex",
                    "SibSp",
                    "Parch",
                    "Fare",
                    "Embarked_Chebourg",
                    "Embarked_Queenstown",
                    "Embarked_Southhampton"
                ]
            ]

            # ---------------- PREDICTION ---------------- #

            prediction = model.predict(
                scaled_df,
                verbose=0
            )[0][0]

            # ---------------- RESULT ---------------- #

            if prediction >= 0.5:

                st.balloons()

                st.markdown(
                    f"""
                    <div class='prediction-box survive'>
                        ✅ SURVIVAL LIKELY<br><br>
                        Confidence: {prediction:.2%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class='prediction-box die'>
                        ❌ SURVIVAL UNLIKELY<br><br>
                        Confidence: {(1 - prediction):.2%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow")