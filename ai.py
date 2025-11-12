import os
import openai
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from fuzzywuzzy import process
from fpdf import FPDF
import re

# Load dataset
file_path = 'symbipredict_2022.csv'
df = pd.read_csv(file_path)

# Convert symptoms to lowercase and clean column names
df.columns = df.columns.str.lower().str.replace('_', ' ')
df.iloc[:, :-1] = df.iloc[:, :-1].applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# Select symptoms (all columns except the last) and disease (last column)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Get all symptoms as a list
all_symptoms = X.columns.tolist()

# Symptom synonyms mapping
symptom_synonyms = {
    "runny nose": "nasal discharge",
    "stomach pain": "abdominal pain",
    "high temperature": "fever",
    "sore throat": "throat pain"
}

# Encode symptoms
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X.apply(lambda row: row[row == 1].index.tolist(), axis=1))

# Encode diseases
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)
disease_mapping = {idx: disease for idx, disease in enumerate(disease_encoder.classes_)}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model
with open('symptom_checker_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('disease_encoder.pkl', 'wb') as f:
    pickle.dump(disease_encoder, f)
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# UI Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        .stTextArea, .stButton {
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #0073e6;
            color: white;
            font-size: 16px;
            padding: 10px;
        }
        .stTitle {
            color: #004d99;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("AI Symptom Checker")
st.write("Describe your symptoms, and our AI will predict possible diseases.")

# Inputs
residence = st.text_input("🏠 Where do you live? (City/Region)")
duration = st.text_input("🕐 How long have you had these symptoms? (e.g., 2 days, 1 week)")
severity = st.selectbox("⚠️ Symptom severity:", ["Mild", "Moderate", "Severe"])
user_input = st.text_area("Enter your symptoms in natural language:")

# Symptom Extraction
def extract_symptoms(text):
    doc = nlp(text.lower())
    extracted = set()
    for token in doc:
        match = process.extractOne(token.text, all_symptoms, score_cutoff=50)
        if match:
            extracted.add(match[0])
        elif token.text in symptom_synonyms:
            extracted.add(symptom_synonyms[token.text])
    return list(extracted)

# Remove non-ASCII
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# PDF Report Generator
def generate_pdf_report(symptoms, diseases, confidences, specialists, severity, duration, residence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def add_line(label, content):
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(40, 10, remove_non_ascii(f"{label}:"), ln=False)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, remove_non_ascii(str(content)))

    # Report Sections
    add_line("User Symptoms", ', '.join(symptoms))
    add_line("Symptom Severity", severity)
    add_line("Symptom Duration", duration)
    add_line("Predicted Diseases", ', '.join(f"{d} ({c:.2f}%)" for d, c in zip(diseases, confidences)))
    add_line("User Residence", residence)

    if specialists:
        for idx, spec in enumerate(specialists, 1):
            add_line(f"Specialist {idx}", f"{spec['name']} ({spec['specialty']})")
            add_line("  Location", spec['location'])
            add_line("  Hospital Address", spec['hospital_address'])
            add_line("  Contact Number", spec['contact_number'])

    file_path = "diagnosis_report.pdf"
    pdf.output(file_path)
    return file_path

# Mock Specialist Data with Details
def get_specialists(diseases, residence):
    if residence.lower() == "new york":
        return [
            {
                "name": "Dr. Smith",
                "specialty": "General Physician",
                "location": "New York",
                "hospital_address": "123 Health St, NY 10001",
                "contact_number": "+1 212-555-1234"
            },
            {
                "name": "Dr. Lee",
                "specialty": "Infectious Diseases",
                "location": "New York",
                "hospital_address": "456 Wellness Ave, NY 10002",
                "contact_number": "+1 212-555-5678"
            }
        ]
    elif residence.lower() == "san francisco":
        return [
            {
                "name": "Dr. Harris",
                "specialty": "Cardiologist",
                "location": "San Francisco",
                "hospital_address": "789 Heart Blvd, SF 94102",
                "contact_number": "+1 415-555-9876"
            },
            {
                "name": "Dr. James",
                "specialty": "Dermatologist",
                "location": "San Francisco",
                "hospital_address": "321 Skin Ln, SF 94103",
                "contact_number": "+1 415-555-4321"
            }
        ]
    else:
        return [
            {
                "name": "Dr. Brown",
                "specialty": "General Physician",
                "location": "Online Consultation",
                "hospital_address": "Virtual Clinic",
                "contact_number": "+1 800-555-0000"
            }
        ]

# Predict Button
if st.button("🔍 Predict Disease"):
    if not user_input.strip():
        st.warning("⚠ Please enter your symptoms.")
    elif not residence.strip():
        st.warning("⚠ Please provide your residence for nearby suggestions.")
    else:
        extracted_symptoms = extract_symptoms(user_input)
        if not extracted_symptoms:
            st.error("❌ No recognizable symptoms found. Try rephrasing.")
        else:
            st.success(f"✅ Extracted Symptoms: {', '.join(extracted_symptoms)}")
            input_encoded = mlb.transform([extracted_symptoms])
            probabilities = model.predict_proba(input_encoded)[0]
            non_zero_indices = [i for i, prob in enumerate(probabilities) if prob > 0.01]
            if not non_zero_indices:
                st.error("❌ No strong matches found.")
            else:
                sorted_indices = sorted(non_zero_indices, key=lambda i: probabilities[i], reverse=True)[:3]
                predicted_diseases = [disease_mapping[idx] for idx in sorted_indices]
                confidences = [probabilities[i] * 100 for i in sorted_indices]

                st.write("### Most Likely Diseases:")
                for i, disease in enumerate(predicted_diseases, start=1):
                    st.write(f"{i}. {disease} - ({confidences[i-1]:.2f}% confidence)")

                specialists = get_specialists(predicted_diseases, residence)
                report_path = generate_pdf_report(extracted_symptoms, predicted_diseases, confidences, specialists, severity, duration, residence)
                st.success("📄 Report generated successfully!")
                with open(report_path, "rb") as f:
                    st.download_button(label="⬇️ Download Report (PDF)", data=f, file_name="diagnosis_report.pdf", mime="application/pdf")
