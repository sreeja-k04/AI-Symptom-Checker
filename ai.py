import pandas as pd
import numpy as np
import pickle
import streamlit as st
import spacy
import openai  # OpenAI integration
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from fuzzywuzzy import process

# 🔑 Set up OpenAI API key (Replace 'your-api-key' with your actual key)
openai.api_key = "your-api-key"

# Function to query OpenAI for answers
def ask_openai(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-4 if available
            messages=[{"role": "system", "content": "You are a medical assistant."},
                      {"role": "user", "content": question}],
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "⚠ OpenAI API Error: " + str(e)

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

# Synonym mapping for symptoms
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Medicine recommendations
medicine_recommendations = {
    "fever": "Paracetamol, Ibuprofen, Hydration",
    "headache": "Paracetamol, Ibuprofen, Rest",
    "cough": "Cough syrup, Warm tea with honey, Lozenges",
    "sore throat": "Saltwater gargle, Lozenges, Ibuprofen",
    "runny nose": "Antihistamines, Steam inhalation",
    "abdominal pain": "Antacids, Peppermint tea, Probiotics"
}

# Doctor recommendation based on disease
doctor_recommendations = {
    "flu": "General Physician",
    "pneumonia": "Pulmonologist",
    "migraine": "Neurologist",
    "gastritis": "Gastroenterologist",
    "bronchitis": "Pulmonologist",
    "covid-19": "Infectious Disease Specialist"
}

# Streamlit UI
st.title("🤖 AI Symptom Checker with OpenAI Chat")
st.write("Describe your symptoms, and our AI will predict possible diseases, suggest medicines, and answer your health-related questions.")

# User Inputs
user_input = st.text_area("Enter your symptoms in natural language:")

# Severity selection
severity = st.selectbox("Select severity:", ["Mild", "Moderate", "Severe"])

# Duration input (in days)
num_days = st.number_input("How many days have you had symptoms?", min_value=1, max_value=30, value=1)

def extract_symptoms(text):
    """Extract and match symptoms from user input."""
    doc = nlp(text.lower())
    extracted = set()
    for token in doc:
        match = process.extractOne(token.text, all_symptoms, score_cutoff=50)
        if match:
            extracted.add(match[0])
        elif token.text in symptom_synonyms:
            extracted.add(symptom_synonyms[token.text])
    return list(extracted)

if st.button("🔍 Predict Disease & Suggest Medicines"):
    if not user_input.strip():
        st.warning("⚠ Please enter your symptoms.")
    else:
        extracted_symptoms = extract_symptoms(user_input)
        if not extracted_symptoms:
            st.error("❌ No recognizable symptoms found. Try rephrasing.")
        else:
            st.success(f"✅ Extracted Symptoms: {', '.join(extracted_symptoms)}")
            
            # Disease Prediction
            input_encoded = mlb.transform([extracted_symptoms])
            probabilities = model.predict_proba(input_encoded)[0]
            non_zero_indices = [i for i, prob in enumerate(probabilities) if prob > 0.01]
            
            if not non_zero_indices:
                st.error("❌ No strong matches found for diseases.")
            else:
                sorted_indices = sorted(non_zero_indices, key=lambda i: probabilities[i], reverse=True)[:3]
                predicted_diseases = [disease_mapping[idx] for idx in sorted_indices]
                
                st.write("### Most Likely Diseases:")
                for i, disease in enumerate(predicted_diseases, start=1):
                    confidence = probabilities[sorted_indices[i-1]] * 100
                    st.write(f"{i}. {disease} - ({confidence:.2f}% confidence)")

                    # If severity is moderate or severe, or symptoms last more than 3 days → Visit Doctor
                    if severity != "Mild" or num_days > 3:
                        doctor = doctor_recommendations.get(disease.lower(), "General Physician")
                        st.write(f"🏥 **Recommendation:** Visit a **{doctor}** for further evaluation.")

            # Medicine Recommendations (Only for mild cases lasting ≤3 days)
            if severity == "Mild" and num_days <= 3:
                st.write("### Suggested Medicines for Your Symptoms:")
                for symptom in extracted_symptoms:
                    if symptom.lower() in medicine_recommendations:
                        st.write(f"💊 *For {symptom}:* {medicine_recommendations[symptom.lower()]}")

# **OpenAI Chat: Ask a Health Question**
st.write("### 💬 Ask AI a Health-Related Question")
user_question = st.text_input("Type your question here:")

if st.button("🤖 Get AI Answer"):
    if user_question.strip():
        ai_answer = ask_openai(user_question)
        st.write("**AI Answer:**", ai_answer)
    else:
        st.warning("⚠ Please enter a question.")
