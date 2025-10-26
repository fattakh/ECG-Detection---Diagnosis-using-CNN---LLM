import os
import pandas as pd
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

from config import PATIENT_LOG, DOCTOR_LOG

def log_data(patient_id, patient_name, doctor_id, doctor_name, file_name, label, conf, llm_out, prescription):
    timestamp = datetime.now().isoformat()
    row = {
        "timestamp": timestamp,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "doctor_id": doctor_id,
        "doctor_name": doctor_name,
        "ecg_file": file_name,
        "cnn_label": label,
        "confidence": f"{conf:.2f}",
        "llm_output": llm_out,
        "prescription": prescription
    }
    df = pd.DataFrame([row])
    for log in [PATIENT_LOG, DOCTOR_LOG]:
        if os.path.exists(log):
            df.to_csv(log, mode='a', header=False, index=False)
        else:
            df.to_csv(log, index=False)

def search_logs(who="Patient", name=""):
    target_log = PATIENT_LOG if who == "Patient" else DOCTOR_LOG
    if not os.path.exists(target_log):
        return pd.DataFrame()
    df = pd.read_csv(target_log)
    col = "patient_name" if who == "Patient" else "doctor_name"
    return df[df[col].str.contains(name, case=False, na=False)]

def export_pdf(patient_name, doctor_name, label, conf, llm_summary, prescription, food_advice):
    path = f"reports/{patient_name.replace(' ', '_')}_ECG_Report.pdf"
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("🩺 ECG DIAGNOSTIC REPORT", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Paragraph(f"Patient: {patient_name}", styles["Normal"]),
        Paragraph(f"Doctor: {doctor_name}", styles["Normal"]),
        Spacer(1, 16),
        Paragraph(f"🔍 CNN Result: {label}", styles["Normal"]),
        Paragraph(f"Confidence: {conf*100:.2f}%", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("🤖 LLM Explanation", styles["Heading2"]),
        Paragraph(llm_summary, styles["Normal"]),
        Spacer(1, 12),
        Paragraph("💊 Suggested Prescription", styles["Heading2"]),
        Paragraph(prescription.replace("\n", "<br />"), styles["Normal"]),
        Spacer(1, 12),
        Paragraph("🥗 Dietary Advice", styles["Heading2"]),
        Paragraph(food_advice.replace("\n", "<br />"), styles["Normal"])
    ]
    doc.build(story)
    return path

def get_food_advice():
    return """
**Recommended Foods:**
- Fresh fruits (apples, oranges, berries)
- Leafy green vegetables (spinach, kale, broccoli)
- Whole grains (oats, brown rice, whole wheat bread)
- Lean proteins (fish, skinless chicken, beans)
- Low-fat dairy (yogurt, skim milk)
- Nuts (almonds, walnuts, in moderation)

**Foods to Avoid:**
- Fried and fatty foods
- Processed meats (sausages, bacon)
- Excess salt and salty snacks
- Sugary drinks and sweets
- Full-fat dairy products
- Excess caffeine and energy drinks
"""