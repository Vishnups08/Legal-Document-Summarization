# Legal Document Summarization

A web application and toolkit for summarizing legal documents using state-of-the-art NLP models. This project enables users to upload legal PDFs and receive concise summaries, with tools for model training, evaluation, and dataset management.

## Features
- **Web Interface**: Upload legal PDF files and receive AI-generated summaries.
- **Custom Model Training**: Fine-tune transformer models (e.g., BART) on your own legal datasets.
- **Evaluation Tools**: Assess summarization quality using ROUGE and BLEU metrics, with visualizations.
- **Extensible Data Pipeline**: Easily add new legal datasets for training and evaluation.

## Project Structure
```
legal/
├── app.py                  # Flask web app for PDF upload and summarization
├── model_training.py       # Script to fine-tune summarization models
├── model_evaluation.py     # Script to evaluate models and visualize metrics
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates (index.html, summary.html)
├── static/                 # Static assets (style.css)
├── data/                   # Legal datasets and preprocessing scripts
│   ├── UK-Abs/             # Example: UK legal data (train/test, summaries, judgments)
│   └── ...
├── evaluation_results/     # Model evaluation outputs (CSV, metrics, charts)
└── ...
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Data**
   - Place your processed legal data in the `data/` directory, following the structure in `UK-Abs`, `IN-Abs`, etc.
   - Ensure you have `train.csv`, `validation.csv`, and `test.csv` in the appropriate subfolders.

## Usage
### 1. Train the Model
Fine-tune the summarization model on your legal dataset:
```bash
python model_training.py
```
- The trained model will be saved in `fine_tuned_model/final_model`.

### 2. Evaluate the Model
Evaluate the model's performance and generate metrics/charts:
```bash
python model_evaluation.py
```
- Results are saved in `evaluation_results/` (CSV, average metrics, and a bar chart).

### 3. Run the Web Application
Start the Flask server to use the web interface:
```bash
python app.py
```
- Visit [http://localhost:5000](http://localhost:5000) in your browser.
- Upload a PDF to receive a summary.

## Data Structure
- **data/UK-Abs/**, **data/IN-Abs/**, etc.:
  - `train-data/`, `test-data/`: Contain `summary/` and `judgement/` text files.
  - `stats-UK-train.txt`, `stats-UK-test.txt`: Data statistics.
- **preprocessing.py**: Script for preparing raw data.

## Evaluation Outputs
- **evaluation_results.csv**: Detailed results for each test case.
- **average_metrics.txt**: Average ROUGE and BLEU scores.
- **metrics_chart.png**: Bar chart of evaluation metrics.

## Customization
- **Model**: Change the base model in `model_training.py` (default: `facebook/bart-large-cnn`).
- **Data**: Add new datasets in the `data/` directory and update scripts as needed.
- **Web UI**: Modify HTML/CSS in `templates/` and `static/` for a custom look.

## Dependencies
- Flask
- PyPDF2
- transformers
- torch
- accelerate
- pandas, numpy, matplotlib, nltk, datasets, rouge_score

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License
MIT License (or specify your own)
