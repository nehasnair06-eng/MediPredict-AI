import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap)');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(20, 184, 166, 0.10),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #f8fafc,
            #eef2ff,
            #f0fdfa
        );
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #172554,
        #0f172a
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #cbd5e1 !important;
    margin-bottom: 30px;
}

/* HERO */
.hero {
    padding: 40px;
    border-radius: 28px;
    background: linear-gradient(
        135deg,
        #111827,
        #312e81,
        #0f766e
    );
    box-shadow: 0 20px 45px rgba(15,23,42,0.16);
    color: white;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero-small {
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a5b4fc;
    font-weight: 700;
}

.hero h1 {
    font-size: 44px;
    margin: 10px 0;
    font-weight: 800;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
    margin: 0;
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.85);
    border-radius: 20px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(15,23,42,0.06);
    border: 1px solid rgba(255,255,255,0.9);
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #312e81;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
}

/* SECTION */
.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #111827;
    margin: 22px 0 18px;
}

/* INFO CARD */
.info-card {
    background: rgba(255,255,255,0.80);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 12px 30px rgba(15,23,42,0.07);
    margin-bottom: 25px;
}

.info-card h3 {
    margin-top: 0;
    color: #111827;
}

.info-card p {
    color: #64748b;
    line-height: 1.6;
}

/* INPUTS */
label {
    font-weight: 600 !important;
    color: #334155 !important;
}

.stNumberInput input {
    border-radius: 12px !important;
    border: 1px solid #dbeafe !important;
    background: white !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(
        135deg,
        #4f46e5,
        #0f766e
    );
    color: white;
    box-shadow: 0 10px 25px rgba(79,70,229,0.25);
    transition: 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(79,70,229,0.35);
}

/* RESULTS */
.result-negative {
    padding: 26px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    box-shadow: 0 10px 25px rgba(15,23,42,0.06);
    margin-top: 20px;
}

.result-negative .result-title {
    font-size: 21px;
    font-weight: 800;
    color: #047857;
    margin-bottom: 8px;
}

.result-positive {
    padding: 26px;
    border-radius: 20px;
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1px solid #fecdd3;
    box-shadow: 0 10px 25px rgba(15,23,42,0.06);
    margin-top: 20px;
}

.result-positive .result-title {
    font-size: 21px;
    font-weight: 800;
    color: #be123c;
    margin-bottom: 8px;
}

.result-text {
    color: #475569;
    font-size: 14px;
    line-height: 1.6;
}

/* DISCLAIMER */
.disclaimer {
    margin-top: 35px;
    padding: 17px 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.70);
    border: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_models():
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
    return diabetes_model, diabetes_scaler

try:
    diabetes_model, diabetes_scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">🩺 MediPredict</div>
        <div class="sidebar-subtitle">AI Healthcare Prediction</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Prediction")
    disease = st.selectbox("Choose condition", ["Diabetes", "Heart Disease"])

    st.markdown("---")
    st.markdown(
        """
        <b>Current Model</b><br>🤖 Machine Learning<br><br>
        <b>Prediction Type</b><br>📊 Risk estimation<br><br>
        <b>Platform</b><br>💻 Streamlit
        """,
        unsafe_allow_html=True
    )

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-small">AI • HEALTH • PREDICTION</div>
        <h1>MediPredict AI</h1>
        <p>Intelligent health-risk prediction powered by machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# METRICS
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">2</div>
            <div class="metric-label">Health Conditions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">3</div>
            <div class="metric-label">ML Algorithms</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">AI</div>
            <div class="metric-label">Prediction Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# DIABETES
# =========================================================
if disease == "Diabetes":
    st.markdown(
        """
        <div class="section-title">🩸 Diabetes Risk Assessment</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Patient Information</h3>
            <p>
                Enter the patient's health measurements below.
                The machine learning model will estimate the
                likelihood of diabetes based on the provided data.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=25)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Analyze Health Data", use_container_width=True):
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]], columns=feature_names)

        input_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_scaled)
        probability = diabetes_model.predict_proba(input_scaled)
        diabetes_probability = probability[0][1]

        st.markdown(
            """
            <div class="section-title">Prediction Analysis</div>
            """,
            unsafe_allow_html=True
        )

        if prediction[0] == 1:
            st.markdown(
                """
                <div class="result-positive">
                    <div class="result-title">⚠️ Higher Predicted Likelihood</div>
                    <div class="result-text">
                        The model identified a higher predicted likelihood of diabetes based on the provided health measurements.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-negative">
                    <div class="result-title">✓ Lower Predicted Likelihood</div>
                    <div class="result-text">
                        The model identified a lower predicted likelihood of diabetes based on the provided health measurements.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Prediction Probability", f"{diabetes_probability * 100:.2f}%")

        with result_col2:
            st.metric("Model Output", "Higher Likelihood" if prediction[0] == 1 else "Lower Likelihood")

        st.progress(float(diabetes_probability))

# =========================================================
# HEART DISEASE
# =========================================================
else:
    st.markdown(
        """
        <div class="section-title">❤️ Heart Disease Prediction</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Heart Health Module</h3>
            <p>
                Heart disease prediction will use cardiovascular measurements and patient medical history.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("❤️ Heart Disease prediction will be added next.")

# =========================================================
# DISCLAIMER
# =========================================================
st.markdown(
    """
    <div class="disclaimer">
        ⚠️ <b>Educational Use Only</b><br><br>
        MediPredict AI provides machine-learning based risk estimates and is not a medical diagnosis.
        Always consult a qualified healthcare professional for medical advice and diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)