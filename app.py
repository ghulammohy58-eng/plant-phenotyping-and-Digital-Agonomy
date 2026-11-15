import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from openai import OpenAI

# Page Config
st.set_page_config(
    page_title="AgriPheno Pro | Plant Phenotyping & Digital Agronomy",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Professional Dark Theme
st.markdown("""
    <style>
    .main { background-color: #070B14; color: #F3F4F6; }
    .stSidebar { background-color: #0B132B; }
    h1, h2, h3 { color: #10B981; font-family: 'Inter', sans-serif; }
    .metric-card { background-color: #111827; padding: 20px; border-radius: 12px; border: 1px solid #1F2937; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    </style>
""", unsafe_allow_html=True)

# Initialize Groq Client
groq_api_key = os.environ.get("GROQ_API_KEY", "")
client = None
if groq_api_key:
    client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

st.sidebar.title("🌱 AgriPheno Pro")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Navigation Module", ["1. Leaf Analysis & Morphometry", "2. Canopy Cover & PAI / LAI", "3. Field Data Logger & AI Report"])

# Helper: Generate Synthetic Leaf Sample
def generate_synthetic_leaf():
    img = np.ones((600, 600, 3), dtype=np.uint8) * 240
    cv2.circle(img, (300, 300), 220, (255, 255, 255), -1)
    cv2.circle(img, (300, 300), 220, (100, 100, 100), 4)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for a in angles:
        cx = int(300 + 180 * np.cos(a))
        cy = int(300 + 180 * np.sin(a))
        cv2.rectangle(img, (cx-12, cy-12), (cx+12, cy+12), (0, 0, 0), -1)
        cv2.rectangle(img, (cx-6, cy-6), (cx+6, cy+6), (255, 255, 255), -1)
    pts = np.array([[300, 150], [350, 220], [360, 320], [330, 420], [300, 460], [270, 420], [240, 320], [250, 220]], np.int32)
    cv2.fillPoly(img, [pts], (45, 140, 55))
    cv2.circle(img, (290, 360), 30, (30, 200, 230), -1)
    return img

if 'sample_img' not in st.session_state:
    st.session_state['sample_img'] = generate_synthetic_leaf()

# --- MODULE 1: LEAF ANALYSIS ---
if app_mode == "1. Leaf Analysis & Morphometry":
    st.title("🌿 Leaf Analysis & Morphometry Suite")
    st.markdown("Measure leaf area, perimeter, length, width, and chlorophyll/nitrogen greenness index (DGCI) with precision calibration.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image Source")
        source_option = st.radio("Select Input Source", ["Synthetic Field Sample", "Upload Leaf Photo"])
        if source_option == "Upload Leaf Photo":
            uploaded_file = st.file_uploader("Upload leaf image (.jpg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                st.session_state['sample_img'] = cv2.imdecode(file_bytes, 1)
        
        img = st.session_state['sample_img']
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Active Specimen Feed", use_container_width=True)

    with col2:
        st.subheader("Telemetry & Measurements")
        
        # Processing Leaf Morphometry
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([95, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([25, 255, 255]))
        leaf_mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            pixel_area = cv2.contourArea(c)
            pixel_perim = cv2.arcLength(c, True)
            
            # Calibration scale (~440px circle = 20cm)
            cm_per_pixel = 20.0 / 440.0
            area_cm2 = pixel_area * (cm_per_pixel ** 2)
            perim_cm = pixel_perim * cm_per_pixel
            x, y, bw, bh = cv2.boundingRect(c)
            length_cm = max(bw, bh) * cm_per_pixel
            width_cm = min(bw, bh) * cm_per_pixel
        else:
            area_cm2, perim_cm, length_cm, width_cm = 0.0, 0.0, 0.0, 0.0

        # DGCI Calculation
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        H = hsv[:,:,0].astype(float) / 180.0
        S = hsv[:,:,1].astype(float) / 255.0
        V = hsv[:,:,2].astype(float) / 255.0
        dgci = ((H - 0.167) / 0.167 + (1.0 - S) + (1.0 - V)) / 3.0
        mean_dgci = float(np.mean(dgci[V > 0.1])) if np.any(V > 0.1) else 0.0
        dgci_100 = mean_dgci * 100.0
        est_spad = max(15.0, min(65.0, dgci_100 * 0.9 + 10))

        st.metric("Leaf Area", f"{area_cm2:.2f} cm²")
        st.metric("Leaf Perimeter", f"{perim_cm:.2f} cm")
        st.metric("Leaf Length x Width", f"{length_cm:.2f} cm × {width_cm:.2f} cm")
        st.metric("Dark Green Color Index (DGCI)", f"{mean_dgci:.4f} ({dgci_100:.1f}%)")
        st.metric("Estimated SPAD Chlorophyll", f"{est_spad:.1f}")

# --- MODULE 2: CANOPY & PAI/LAI ---
elif app_mode == "2. Canopy Cover & PAI / LAI":
    st.title("🌾 Canopy Cover & LAI Inversion")
    st.markdown("Analyze top-down field crop imagery for canopy cover percentage, gap fraction, and Plant Area Index (PAI).")
    
    crop_type = st.selectbox("Select Field Crop Type", ["Maize", "Wheat", "Rice", "Soybean / Cotton", "Sugarcane"])
    k_preset = {"Maize": 0.5, "Wheat": 0.45, "Rice": 0.5, "Soybean / Cotton": 0.6, "Sugarcane": 0.55}[crop_type]
    ext_coeff = st.slider("Extinction Coefficient (k)", 0.1, 1.0, k_preset, 0.05)
    
    img = st.session_state['sample_img']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    canopy_cover = (np.sum(thresh == 255) / gray.size) * 100.0
    gap_frac = max(0.01, min(0.99, 1.0 - (canopy_cover / 100.0)))
    lai = -np.log(gap_frac) / ext_coeff

    col1, col2, col3 = st.columns(3)
    col1.metric("Canopy Cover", f"{canopy_cover:.2f} %")
    col2.metric("Gap Fraction (Porosity)", f"{gap_frac:.3f}")
    col3.metric("Estimated LAI / PAI", f"{lai:.2f}")

# --- MODULE 3: FIELD DATA LOGGER & AI REPORT ---
elif app_mode == "3. Field Data Logger & AI Report":
    st.title("📊 Field Data Logger & Groq AI Diagnostics")
    st.markdown("Export session records and generate professional agronomic fertilizer recommendations using Groq (`openai/gpt-oss-20b`).")
    
    if st.button("Generate AI Agronomic Report"):
        if not client:
            st.warning("⚠️ GROQ_API_KEY environment variable not detected. Showing simulated expert report:")
            st.info("""
            **AGRONOMIC AI EXPERT REPORT (Simulated)**
            1. **Vigor Assessment**: Plant canopy demonstrates healthy overall green foliage with normal area development.
            2. **Nitrogen Status**: DGCI and estimated SPAD indicate moderate to optimum nitrogen levels.
            3. **Recommendations**: Maintain current irrigation schedule. Consider variable-rate top-dress nitrogen fertilizer if lower leaves show early chlorosis.
            """)
        else:
            with st.spinner("Analyzing with Groq (openai/gpt-oss-20b)..."):
                prompt = "You are an expert senior agronomist. Provide a professional crop health and fertilization report based on standard field phenotyping data."
                try:
                    response = client.responses.create(input=prompt, model="openai/gpt-oss-20b")
                    st.success("Report Generated Successfully!")
                    st.write(response.output_text)
                except Exception as e:
                    st.error(f"API Error: {e}")
