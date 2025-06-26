import os
from flask import Flask, render_template, request, redirect, url_for, flash
import PyPDF2
from transformers import pipeline
import torch

app = Flask(__name__)
app.secret_key = "legal_summarizer_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

FINE_TUNED_MODEL_PATH = "./model_training"
if os.path.exists(FINE_TUNED_MODEL_PATH):
    print(f"Loading fine-tuned model from {FINE_TUNED_MODEL_PATH}")
    summarizer = pipeline("summarization", model=FINE_TUNED_MODEL_PATH, framework="pt")
else:
    print("Using pre-trained model facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def summarize_text(text, max_length=1024):
    max_chunk_length = 900  
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    
    for chunk in chunks:
        if len(chunk) < 50:  
            continue
        
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            if summary and len(summary) > 0:
                summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            continue
    
    final_summary = " ".join(summaries)
    
    if len(final_summary) > max_length:

        final_summary = final_summary[:max_length] + "... (Summary truncated due to length)"
    
    return final_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
            
            if not text or len(text) < 50:
                flash('Could not extract sufficient text from the PDF')
                return redirect(request.url)
            
            # Summarize the extracted text
            summary = summarize_text(text)
            
            return render_template('summary.html', filename=file.filename, summary=summary, original_text=text)
        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return redirect(request.url)
    
    flash('Only PDF files are allowed')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)