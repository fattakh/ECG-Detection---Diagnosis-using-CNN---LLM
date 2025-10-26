import os

# Set up directories
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# File paths
PATIENT_LOG = "logs/patient_log.csv"
DOCTOR_LOG  = "logs/doctor_log.csv"
MODEL_PATH  = "models/ecg_resnet18.pth"

# Dataset
DATASET_PATH = 'ECG_DATA'
CLASS_NAMES = [
    "Atrial Fibrillation",
    "Ventricular Tachycardia",
    "Myocardial Infarction",
    "Bundle Branch Block"
]