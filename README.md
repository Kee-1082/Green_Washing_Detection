# 🌿 GreenWashDet: Explainable NLP for Corporate Greenwashing Detection

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GreenWashDet** is an end-to-end Explainable Natural Language Processing (xNLP) framework and web application designed to detect and classify environmental claims in corporate sustainability reports (e.g., BRR/BRSR filings). 

Using interpretable linguistic feature extraction combined with machine learning (Multinomial Naive Bayes), GreenWashDet parses corporate PDF reports, segments environmental statements, and categorizes them into **Vague**, **Substantive**, or **Numeric-Financial** claims—providing full transparency into model predictions and visual analytics.

---

## 📸 Overview & Key Features

- 📑 **Automated PDF Report Parsing**: Extracts and segments environmental statements from heavy corporate filings (BRR/BRSR reports) using fast, reliable regex and text extraction (`pdfplumber`).
- 🧠 **Explainable NLP (xNLP) Feature Engineering**: Uses 5 rule-based, interpretable linguistic features instead of opaque black-box embeddings:
  1. **Vague Adjective Ratio**: Proportion of hedging/aspirational adjectives (e.g., *aim, strive, endeavour, committed*).
  2. **Quantifier Count**: Presence of measurable metrics (e.g., *%, tonnes, tCO2, MW, kL, m³*).
  3. **Verb Strength**: Classification of active/past verifiable verbs vs. future/modal aspirational verbs.
  4. **Target Year Presence**: Identification of specific target/baseline years (e.g., *2030, FY2023*).
  5. **Specific Technology Mentions**: Term matching for concrete green tech (e.g., *CCUS, waste heat recovery, solar PV*).
- 📊 **Interactive Dark-Themed Web Dashboard**: Built with Flask and a modern Glassmorphism UI, offering:
  - Real-time sentence-by-sentence classification & confidence scoring.
  - Interactive distribution charts, claim proportions, and top feature importance breakdowns generated dynamically via `matplotlib`.
  - Sentence filtering by claim type (Vague, Substantive, Numeric).
- 🔬 **Academic Research Pipeline**:
  - Hybrid annotation framework (manual annotation + xNLP active expansion + low-confidence manual QA).
  - Inter-Annotator Agreement (IAA) calculation using Cohen's Kappa.
  - Script for generating publication-ready research figures.

---

## 🏷️ Annotation & Classification Schema

Sentences extracted from corporate reports are classified into three distinct categories based on explicit annotation guidelines:

| Label | Name | Definition & Characteristics | Example |
| :---: | :--- | :--- | :--- |
| **`V`** | **Vague** | Aspirational/marketing language with hedging terms and **no measurable specifics** (no numbers, targets, or named tech). | *"We are committed to reducing our environmental impact and striving for a greener future."* |
| **`S`** | **Substantive** | Specific, quantified, or verifiable claim containing **at least one concrete metric** (number, %, target year, named technology, baseline). | *"We reduced specific CO2 emissions by 15% in FY2023 compared to the FY2019 baseline."* |
| **`N`** | **Numeric-Financial** | Tabular data rows, metrics, or financial line items with minimal surrounding prose. | *"Scope 1 Emissions: 1,420,000 tCO2e (FY2022-23)."* |

---

## 🛠️ Tech Stack

### **Backend & Machine Learning**
- **Python 3.9+**
- **Flask**: Web application server and API endpoints.
- **scikit-learn**: Model training (`MultinomialNB`), feature scaling (`MinMaxScaler`), cross-validation.
- **pdfplumber**: PDF text extraction and document parsing.
- **NumPy & Pandas**: Data manipulation and feature matrix operations.

### **Frontend & Visual Analytics**
- **HTML5 & CSS3**: Custom dark-themed Glassmorphism aesthetic.
- **Vanilla JavaScript**: Asynchronous API interactions (Fetch API) and UI responsiveness.
- **Matplotlib**: Dynamic base64 figure generation for inline visual analytics (distribution, feature importances, sentence lengths).

---

## 📁 Repository Structure

```text
GreenWashDet/
├── app.py                      # Flask web server & real-time inference endpoint
├── requirements.txt            # Python dependencies
├── annotations/                # Datasets & annotation guidelines
│   ├── annotation_guidelines.md # Labeling criteria (V, S, N)
│   ├── full_corpus_labeled.csv  # Final labeled dataset (1,172 sentences)
│   └── low_confidence_review.csv# Flagged predictions for manual QA
├── scripts/                    # Core NLP pipeline & research scripts
│   ├── feature_extractor.py    # 5 explainable linguistic feature extractors
│   ├── sentence_segmenter.py   # PDF text extraction & sentence segmentation
│   ├── train_predict.py        # Naive Bayes training & corpus prediction
│   ├── generate_figures.py     # Publication figure generator
│   ├── hybrid_kappa.py         # Inter-Annotator Agreement (Cohen's Kappa)
│   ├── column_extractor.py     # PDF column-aware text extractor
│   └── xnlp_annotator.py       # Interactive xNLP annotation tool
├── templates/
│   └── index.html              # Modern dark-mode web dashboard UI
├── static/                     # CSS & client-side assets
├── outputs/                    # Exported figures and generated charts
├── data/                       # Raw input PDFs and corpora (ignored by git)
└── logs/                       # Training and processing logs
```

---

## 🚀 Getting Started

Follow these steps to run GreenWashDet locally on your machine.

### 📋 Prerequisites
- Python **3.9** or higher installed on your system.
- Git.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Kee-1082/Green_Washing_Detection.git
cd Green_Washing_Detection
```

### 2️⃣ Create and Activate a Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Web Application

Launch the Flask development server:

```bash
python app.py
```

Once started, open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### 🔍 How to Use the Web App:
1. **Upload Report**: Select a corporate sustainability report (PDF file).
2. **Run Analysis**: Click **Analyze Document** to trigger PDF sentence extraction, xNLP feature scoring, and model inference.
3. **View Dashboard**:
   - **Executive Metrics**: See total environmental sentences, Vague vs. Substantive ratio, and confidence scores.
   - **Visual Analytics**: Interactive dark-themed charts showing class distributions, confidence breakdown, feature salience, and sentence length impact.
   - **Detailed Sentence Table**: Inspect individual sentences with color-coded badges (`Vague`, `Substantive`, `Numeric`) and full feature breakdowns.

---

## 🔬 Running Pipeline Scripts

You can also execute individual pipeline modules from the command line:

### 1. Model Training & Active Corpus Prediction
Train the Naive Bayes model on labeled seed data, predict remaining corpus sentences, and flag low-confidence samples (probability < 0.65):
```bash
python scripts/train_predict.py
```
To set a custom confidence threshold (e.g., 0.70):
```bash
python scripts/train_predict.py --threshold 0.70
```

### 2. Generate Academic Figures
Generate all publication-ready research plots (confusion matrices, feature distributions, class balances):
```bash
python scripts/generate_figures.py
```
*Output images will be saved in the `outputs/` directory.*

### 3. Compute Inter-Annotator Agreement (IAA)
Calculate Cohen's Kappa across annotator sessions:
```bash
python scripts/hybrid_kappa.py
```

---

## 📄 License & Citation

This project is developed for research into Explainable Natural Language Processing for Corporate Sustainability Reporting.

If you find this project or dataset helpful in your research, please consider citing:
```bibtex
@article{GreenWashDet2026,
  title={Explainable Natural Language Processing for Corporate Greenwashing Detection},
  author={GreenWashDet Team},
  year={2026}
}
```
