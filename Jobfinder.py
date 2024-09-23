import torch
from transformers import AutoModel, AutoTokenizer
import spacy
from spacy import displacy
import numpy as np
from PyPDF2 import PdfFileReader
from pdf2text import pdf2text
from flask import Flask, request, jsonify


app = Flask(__name__)


# Load pre-trained model and tokenizer
model_name = "t5-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load spacy model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(file_path):
    # Extract text from PDF
    text = pdf2text(file_path)
    return text


def extract_main_points(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate key points
    outputs = model.generate(inputs["input_ids"], num_beams=4, no_repeat_ngram_size=3, min_length=30, max_length=100)

    # Convert generated IDs to text
    main_points = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return main_points


def extract_resume_info(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ == "NOUN"]
    experience = [token.text for token in doc if token.pos_ == "VERB"]
    education = [token.text for token in doc if token.pos_ == "PROPN"]
    projects = []
    
    # Extract project information
    for sentence in doc.sents:
        if "project" in sentence.text.lower():
            project = {}
            for token in sentence:
                if token.pos_ == "NOUN":
                    project["name"] = token.text
                elif token.pos_ == "VERB":
                    project["description"] = sentence.text
            projects.append(project)
    
    return skills, experience, education, projects


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = file.filename
    file.save(file_path)
    
    # Extract text from PDF
    text = extract_text_from_pdf(file_path)
    
    # Extract main points and resume info
    main_points = extract_main_points(text)
    skills, experience, education, projects = extract_resume_info(text)
    
    # Return extracted information as JSON
    return jsonify({
        'main_points': main_points,
        'skills': skills,
        'experience': experience,
        'education': education,
        'projects': projects
    })


if __name__ == '__main__':
    app.run(debug=True)