

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pypdfium2
import fitz
import os

model = AutoModelForSequenceClassification.from_pretrained("REEM-ALRASHIDI/LongFormer-Paper-Citaion-Classifier")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def extract_text_from_pdf(file_path):
    text = ''
    with fitz.open(file_path) as pdf_document:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text += page.get_text()
    return text

def predict_class(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

uploaded_files_dir = "uploaded_files"
os.makedirs(uploaded_files_dir, exist_ok=True)

st.title("Paper Citation Classifier")

option = st.radio("Select input type:", ("Text", "PDF"))

if option == "Text":
    text_input = st.text_area("Enter your text here:")
    if st.button("Predict") and text_input.strip():
        predicted_class = predict_class(text_input)
        class_labels = ["Level 1", "Level 2", "Level 3", "Level 4"]
        st.text(f"Predicted Class: {class_labels[predicted_class]}")
        
elif option == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file (Max size: 200 MB)", type=["pdf"])
    if uploaded_file is not None:
        file_path = os.path.join(uploaded_files_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully.")
        st.text(f"File Path: {file_path}") 
        file_text = extract_text_from_pdf(file_path)
        st.text("Extracted Text:")
        st.text(file_text)
        
        if st.button("Predict"):
            predicted_class = predict_class(file_text)
            class_labels = ["Level 1", "Level 2", "Level 3", "Level 4"]
            st.text(f"Predicted Class: {class_labels[predicted_class]}")
