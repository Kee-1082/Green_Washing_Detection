import pdfplumber
import os

input_root = "data/raw"
output_root = "data/extracted"


def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def write_output(company, year, text):
    out_dir = os.path.join(output_root, company)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{year}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done: {company}/{year}")

for company in os.listdir(input_root):
    company_path = os.path.join(input_root, company)

    if os.path.isdir(company_path):
        for filename in os.listdir(company_path):
            if filename.endswith(".pdf"):
                year = filename.replace(".pdf", "")
                pdf_path = os.path.join(company_path, filename)
                text = extract_pdf_text(pdf_path)
                write_output(company, year, text)
    elif company.endswith(".pdf"):
        # Supports flat input files like: raw_jsw_2025.pdf
        stem = os.path.splitext(company)[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0].lower() == "raw":
            company_name = "_".join(parts[1:-1])
            year = parts[-1]
        else:
            company_name = "misc"
            year = stem

        text = extract_pdf_text(company_path)
        write_output(company_name, year, text)