"""
Disease Classification System - CEO-Level Dark Glassmorphic Medical Dashboard
Developed by: Easy Business Technology (Home of Project, Research and Mentorship)
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. PAGE SETUP & CEO-LEVEL DARK GLASSMORPHISM CSS
# ==========================================
st.set_page_config(
    page_title="Disease Classification System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Hospital Dashboard Dark UI Template
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Core Layout Configurations */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F1F5F9;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F172A 0%, #020617 100%);
    }

    /* Cinematic Smooth Fade-In Animations */
    @keyframes subtleFadeUp {
        0% { opacity: 0; transform: translateY(12px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes subtleGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(14, 165, 233, 0.15); }
        50% { box-shadow: 0 0 25px rgba(14, 165, 233, 0.35); }
    }

    /* Executive Hero Header Banner */
    .executive-hero {
        background: linear-gradient(135deg, rgba(3, 105, 161, 0.2) 0%, rgba(15, 23, 42, 0.4) 100%);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        color: #FFFFFF;
        border: 1px solid rgba(14, 165, 233, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 2.5rem;
        animation: subtleFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .hero-title { font-size: 2.5rem; font-weight: 700; margin: 0; color: #38BDF8; letter-spacing: -0.5px; }
    .hero-subtitle { font-size: 1.15rem; color: #94A3B8; margin-top: 10px; font-weight: 400; }

    /* Ultra-Premium Glassmorphism Intake Cards */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        animation: subtleFadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Individual Parameter Titles */
    .feature-title-block {
        font-size: 1.25rem;
        font-weight: 600;
        color: #38BDF8;
        margin-top: 2.2rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-left: 4px solid #0EA5E9;
        padding-left: 10px;
    }
    
    /* High-Level Diagnostic Output Presentation Cards */
    .premium-probability-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 20px;
        padding: 2.2rem;
        border: 1px solid #38BDF8;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        animation: subtleFadeUp 0.5s ease-out forwards, subtleGlow 4s infinite ease-in-out;
    }

    /* Highly Responsive Flex-Grid Action Buttons for Production layouts */
    div[data-testid="stForm"]  .stButton > button {
        width: 100% !important;
        border-radius: 40px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 10px;
    }
    div[data-testid="stForm"]  .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.35);
    }
    div[data-testid="stForm"]  .stButton > button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.55);
    }
    
    /* Premium Sidebar Fine Tuning */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070A12 0%, #020408 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    /* Audit Log Component Items */
    .sidebar-audit-log-item {
        padding: 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    
    /* Legal Disclaimer & Footer Configurations */
    .premium-disclaimer {
        background: rgba(217, 119, 6, 0.08);
        border: 1px solid rgba(217, 119, 6, 0.15);
        border-left: 4px solid #F59E0B;
        padding: 1.5rem;
        border-radius: 16px;
        color: #FDE047;
        font-size: 0.92rem;
        margin-top: 4.5rem;
        line-height: 1.6;
    }
    .premium-footer {
        text-align: center;
        color: #475569;
        font-size: 0.88rem;
        margin-top: 4.5rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        letter-spacing: 0.3px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Configurations
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ==========================================
# 2. MODEL LAYER LOAD PROCESSOR
# ==========================================
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('diseases_classifications.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except Exception as e:
        st.error(f"🛑 Error loading operational medical assets: {e}")
        return None, None

model, label_encoder = load_ml_assets()

# ==========================================
# 3. PREMIUM DESIGNED SIDEBAR PLATFORM
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#38BDF8; font-size:1.4rem; font-weight:700; margin-bottom:0;'>🩺 Clinical Hub</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:0.8rem; margin-top:4px;'>Disease Classification System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### ⚙️ Operational Telemetry")
    st.markdown("""
    * **Algorithm Base:** Logistic Regression
    * **Analytics Type:** Supervised Classification
    * **Pipeline Profile:** Binary Target System
    * **Deployment Core:** Version 1.0
    """)
    st.markdown("---")
    
    st.markdown("#### 🕒 Operational Audit Trail (History)")
    if not st.session_state.prediction_history:
        st.caption("No predictive operations executed in this tracking session.")
    else:
        for trace in reversed(st.session_state.prediction_history):
            st.markdown(f"""
            <div class="sidebar-audit-log-item">
                <span style="color:#64748B;">⏱️ {trace['time']}</span><br>
                Diagnosis: <span style="color:#38BDF8; font-weight:600;">{trace['disease']}</span><br>
                Confidence Level: <b>{trace['conf']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("""
    <div style='background: rgba(14, 165, 233, 0.05); padding: 16px; border-radius: 16px; border: 1px solid rgba(14, 165, 233, 0.15);'>
        <span style='color: #64748B; font-weight: bold; font-size: 0.75rem; text-transform: uppercase;'>SYSTEM ARCHITECT:</span><br>
        <span style='color: #38BDF8; font-weight: 700; font-size: 1rem;'>Easy Business Technology</span><br>
        <span style='color: #94A3B8; font-size: 0.75rem; display:block; margin-top:2px;'>Home of Project, Research and Mentorship</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. CHIEF EXECUTIVE OFFICER HERO HEADER
# ==========================================
st.markdown("""
    <div class="executive-hero">
        <h1 class="hero-title">🧠 Disease Classification Intelligence Platform</h1>
        <p class="hero-subtitle">High-fidelity binary evaluation system driven by custom pre-trained Logistic Regression models.</p>
    </div>
""", unsafe_allow_html=True)

if model is None or label_encoder is None:
    st.stop()

# Layout Partition Setup
col_form, col_results = st.columns([1.25, 1], gap="large")

# ==========================================
# 5. FORM LAYER WITH DEDICATED BLOCK LABELS
# ==========================================
with col_form:
    st.markdown("<h3 style='font-weight:600; color:#FFFFFF; margin-bottom:1rem;'>📋 Automated Clinical Intake Registry</h3>", unsafe_allow_html=True)
    
    with st.form(key='executive_clinical_form'):
        
        # --- Age ---
        st.markdown('<p class="feature-title-block">🔢 Patient Chronological Age</p>', unsafe_allow_html=True)
        age = st.number_input("Input the raw age of the patient in whole calendar years", min_value=0, max_value=120, value=25, step=1, key="age")
        
        # --- Gender ---
        st.markdown('<p class="feature-title-block">🧬 Assigned Biological Gender</p>', unsafe_allow_html=True)
        gender_raw = st.selectbox("Specify phenotypic expression marker at registration", options=["Male", "Female"], key="gender")
        
        # --- Fever ---
        st.markdown('<p class="feature-title-block">🔥 Systemic Fever Symptom</p>', unsafe_allow_html=True)
        fever_raw = st.selectbox("Is the patient presenting with an elevated body baseline temperature?", options=["No", "Yes"], key="fever")
        
        # --- Headache ---
        st.markdown('<p class="feature-title-block">🧠 Cranial Headache Severity</p>', unsafe_allow_html=True)
        headache_raw = st.selectbox("Are persistent or acute headaches actively reported?", options=["No", "Yes"], key="headache")
        
        # --- Body Pain ---
        st.markdown('<p class="feature-title-block">💪 Myalgia & Generalized Body Pain</p>', unsafe_allow_html=True)
        body_pain_raw = st.selectbox("Is systemic muscular or skeletal pain present?", options=["No", "Yes"], key="body_pain")
        
        # --- Vomiting ---
        st.markdown('<p class="feature-title-block">🤢 Emesis / Vomiting Tendencies</p>', unsafe_allow_html=True)
        vomiting_raw = st.selectbox("Have any vomiting episodes occurred during the current sequence?", options=["No", "Yes"], key="vomiting")
        
        # --- Weakness ---
        st.markdown('<p class="feature-title-block">🔋 Acute Physiological Weakness</p>', unsafe_allow_html=True)
        weakness_raw = st.selectbox("Is profound lethargy or bodily weakness observable?", options=["No", "Yes"], key="weakness")
        
        # --- Chills ---
        st.markdown('<p class="feature-title-block">🥶 Rigors & Involuntary Chills</p>', unsafe_allow_html=True)
        chills_raw = st.selectbox("Are sudden cold flashes or shivering episodes active?", options=["No", "Yes"], key="chills")
        
        # --- Stomach Pain ---
        st.markdown('<p class="feature-title-block">🤢 Abdominal Distress / Stomach Pain</p>', unsafe_allow_html=True)
        stomach_pain_raw = st.selectbox("Is localized gastrointestinal/stomach discomfort reported?", options=["No", "Yes"], key="stomach_pain")
        
        # --- Diarrhea ---
        st.markdown('<p class="feature-title-block">🧻 Diarrhea Manifestation</p>', unsafe_allow_html=True)
        diarrhea_raw = st.selectbox("Is the patient experiencing hyper-frequent watery stool anomalies?", options=["No", "Yes"], key="diarrhea")
        
        # --- Loss of Appetite ---
        st.markdown('<p class="feature-title-block">🍽️ Loss of Appetite (Anorexia)</p>', unsafe_allow_html=True)
        loss_appetite_raw = st.selectbox("Has nutritional ingestion capacity fallen due to low appetite?", options=["No", "Yes"], key="loss_of_appetite")
        
        # --- Body Temperature ---
        st.markdown('<p class="feature-title-block">🌡️ Core Body Temperature (°C)</p>', unsafe_allow_html=True)
        body_temp = st.number_input("Log exact calibrated metric thermometer readout value", min_value=30.0, max_value=45.0, value=37.0, step=0.1, key="body_temperature")
        
        # --- Duration of Days ---
        st.markdown('<p class="feature-title-block">⏳ Symptom Persistence Duration</p>', unsafe_allow_html=True)
        duration_days = st.number_input("Define the total integer count of days since manifestation onset", min_value=0, max_value=365, value=1, step=1, key="duration_days")

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
        
        # Fully Responsive Fluid Button Row Setup
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            submit_btn = st.form_submit_button(label="Analyze Baseline Diagnostics")
        with btn_col2:
            clear_btn = st.form_submit_button(label="Flush Data Fields")

    if clear_btn:
        st.session_state.form_submitted = False
        st.rerun()

    if submit_btn:
        st.session_state.form_submitted = True

# ==========================================
# 6. EXHAUSTIVE ML PROCESSING & PROBABILITY CARDS
# ==========================================
with col_results:
    st.markdown("<h3 style='font-weight:600; color:#FFFFFF; margin-bottom:1rem;'>🎯 Analytical Outputs & Projections</h3>", unsafe_allow_html=True)
    
    if st.session_state.form_submitted:
        with st.spinner("Processing structural values via array normalization layers..."):
            
            # Strict Preprocessing Maps
            gender = 0 if gender_raw == "Male" else 1
            fever = 1 if fever_raw == "Yes" else 0
            headache = 1 if headache_raw == "Yes" else 0
            body_pain = 1 if body_pain_raw == "Yes" else 0
            vomiting = 1 if vomiting_raw == "Yes" else 0
            weakness = 1 if weakness_raw == "Yes" else 0
            chills = 1 if chills_raw == "Yes" else 0
            stomach_pain = 1 if stomach_pain_raw == "Yes" else 0
            diarrhea = 1 if diarrhea_raw == "Yes" else 0
            loss_of_appetite = 1 if loss_appetite_raw == "Yes" else 0

            # Pre-prediction Dataframe Layout Alignment Assembly
            raw_input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Fever': fever,
                'Headache': headache,
                'Body_Pain': body_pain,
                'Vomiting': vomiting,
                'Weakness': weakness,
                'Chills': chills,
                'Stomach_Pain': stomach_pain,
                'Diarrhea': diarrhea,
                'Loss_of_Appetite': loss_of_appetite,
                'Body_Temperature': body_temp,
                'Duration_Days': duration_days
            }])
            
            # Enforcing feature sequence continuity
            structured_order = [
                'Age', 'Gender', 'Fever', 'Headache', 'Body_Pain', 'Vomiting', 
                'Weakness', 'Chills', 'Stomach_Pain', 'Diarrhea', 'Loss_of_Appetite', 
                'Body_Temperature', 'Duration_Days'
            ]
            final_inference_df = raw_input_data[structured_order]

            # Inferences Ingestion Block
            prediction_array = model.predict(final_inference_df)
            predicted_disease = label_encoder.inverse_transform(prediction_array)[0]
            
            probabilities = model.predict_proba(final_inference_df)[0]
            classes = label_encoder.classes_
            probability_matrix_map = dict(zip(classes, probabilities))
            target_class_confidence = probability_matrix_map.get(predicted_disease, 0.0) * 100

            # Log execution events into system track cache
            st.session_state.prediction_history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'disease': predicted_disease,
                'conf': target_class_confidence
            })

        # Display CEO-Level Premium Glowing Evaluation Card
        st.markdown(f"""
            <div class="premium-probability-card">
                <span style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;">Primary Classification Output</span>
                <h2 style="color: #38BDF8; margin: 6px 0 10px 0; font-weight: 700; font-size: 2.5rem;">🩺 {predicted_disease}</h2>
                <span style="font-size: 1.05rem; color: #E2E8F0;">System Classification Certainty Index: <strong style="color: #38BDF8;">{target_class_confidence:.1f}%</strong></span>
            </div>
        """, unsafe_allow_html=True)
        
        st.success("📊 Classification sequence executed safely against parameters.")
        
        # Display Probability Progress Cards 
        st.markdown("<p style='font-weight: 600; margin-top: 2rem; color:#F1F5F9; font-size: 1.1rem;'>Full Diagnostics Probability Variance Profiles:</p>", unsafe_allow_html=True)
        
        for name, probability_value in sorted(probability_matrix_map.items(), key=lambda elem: elem[1], reverse=True):
            percentage = probability_value * 100
            st.markdown(f"""
                <div style='display:flex; justify-content:space-between; font-size:0.95rem; margin-top:14px; color:#94A3B8;'>
                    <span>🔬 Clinical Class Target: <b>{name}</b></span>
                    <span style='color:#38BDF8;'><b>{percentage:.1f}%</b></span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(float(probability_value))
            
    else:
        st.markdown("""
            <div style="text-align: center; border: 2px dashed rgba(56, 189, 248, 0.15); padding: 6rem 2rem; border-radius: 20px; background-color: rgba(255,255,255,0.01);">
                <span style="font-size: 3.5rem; color: #0EA5E9; opacity: 0.5;">🧬</span>
                <h4 style="color: #94A3B8; margin-top: 1.5rem; font-weight: 500; font-size:1.1rem;">Awaiting Diagnostic Signal</h4>
                <p style="color: #475569; font-size: 0.88rem; max-width: 340px; margin: 8px auto 0 auto; line-height: 1.6;">Fill out the patient configuration matrix variables inside the intake model grid and trigger core evaluation analytics loops.</p>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. REGULATORY DISCLAIMER & EXECUTIVE FOOTER
# ==========================================
st.markdown("""
    <div class="premium-disclaimer">
        <strong>🛡️ MANDATORY RECONNAISSANCE NOTICE:</strong> This AI prediction is intended for educational and research purposes only. It must not replace laboratory testing or consultation with qualified healthcare professionals.
    </div>
    
    <div class="premium-footer">
        <strong>Developed by Update Abdullahi
        </strong><br>
        <span style="color:#475569; font-size:0.8rem;">Sponsored by Easy Business Technology Nigeria LTD.</span>
    </div>
""", unsafe_allow_html=True)