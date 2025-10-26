import streamlit as st
from PIL import Image
import os

from cnn_model import train_cnn_model, load_cnn, transform_image, preprocess_image
from llm_reasoner import extract_features, build_prompt, load_llm
from utils_backend import log_data, export_pdf, get_food_advice, search_logs
from config import CLASS_NAMES

st.set_page_config("🩺 ECG Diagnosis Expert AI", layout="wide")
st.title("🩺 Dual-AI ECG Diagnosis (CNN + LLM)")

with st.spinner("Checking/Training CNN model..."):
    try:
        train_cnn_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

with st.form("input_info"):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name")
        patient_id = st.text_input("Patient ID")
    with col2:
        doctor_name = st.text_input("Doctor Name")
        doctor_id = st.text_input("Doctor ID")
    uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"])
    submitted = st.form_submit_button("Diagnose")

if submitted and uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG", use_container_width=True)

    binary_img, color_img = preprocess_image(image)
    st.image(binary_img, caption="Binarized ECG", use_container_width=True)

    model = load_cnn()
    tensor = transform_image(color_img)
    pred_idx, confidence = model.predict(tensor)
    label = CLASS_NAMES[pred_idx]
    st.success(f"Prediction: **{label}** ({confidence*100:.2f}%)")

    features = extract_features(binary_img)
    prompt = build_prompt(features)
    llm = load_llm()
    llm_output = llm.query(prompt)
    st.info(f"LLM Suggestion: {llm_output}")


    prescription = "Aspirin 75mg daily\nBeta-blocker if indicated\nSchedule follow-up ECG in 3 days"
    st.markdown("### 💊 Suggested Prescription")
    st.markdown(prescription)

    food_advice = get_food_advice()
    st.markdown("### 🥗 Dietary Advice")
    st.markdown(food_advice)

    log_data(patient_id, patient_name, doctor_id, doctor_name, uploaded_file.name, label, confidence, llm_output, prescription)

    pdf_path = export_pdf(patient_name, doctor_name, label, confidence, llm_output, prescription, food_advice)
    with open(pdf_path, "rb") as f:
        st.download_button("⬇ Download PDF", f, file_name=os.path.basename(pdf_path))

st.markdown("---")
st.subheader("🔍 Search Diagnosis Logs")
log_option = st.radio("Search by", ["Patient", "Doctor"])
search_query = st.text_input(f"{log_option} Name")
if st.button("Search Logs"):
    logs = search_logs(log_option, search_query)
    st.dataframe(logs)