import os

input_root = "data/extracted"
output_root = "data/brsr_sections"

START_KEYWORDS = [
    "business responsibility report",
    "sustainability report",
    "business responsibility and sustainability report",
    "brsr",
    "brr",
    "Section A: General Disclosures",
    "integrated report",
    "sustainability",
    "environmental, social and governance",
    "esg focus areas",
    "materiality",
    "principle 1",
    "principle 2",
    "principle 3",
    "principle 4",
    "principle 5",
    "principle 6",
    "principle 7",
    "principle 8",
    "principle 9",
    "essential indicators"
]

END_KEYWORDS = [
    "independent auditor",
    "standalone balance sheet",
    "standalone financial statements",
    "financial statements",
    "notes to accounts",
    "notes to financial statements",
    "report on corporate governance",
    "board's report",
    "directors' report",
    "directors and key management personnel",
    "scope and boundary ",
    "annexure",
    "annexure - a to directors' report",
    "management discussion and analysis",
    "notes to the consolidated financial statements"
]

def find_section(text):
    lines = text.split("\n")
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        lower = line.lower().strip()
        if start_idx is None:
            if any(kw in lower for kw in START_KEYWORDS):
                start_idx = i
        else:
            if any(kw in lower for kw in END_KEYWORDS):
                end_idx = i
                break

    if start_idx is None:
        return None  # section not found

    end_idx = end_idx or len(lines)
    return "\n".join(lines[start_idx:end_idx])

for company in os.listdir(input_root):
    company_path = os.path.join(input_root, company)
    for filename in os.listdir(company_path):
        if filename.endswith(".txt"):
            year = filename.replace(".txt", "")
            with open(os.path.join(company_path, filename), "r", encoding="utf-8") as f:
                text = f.read()

            section = find_section(text)

            out_dir = os.path.join(output_root, company)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{year}.txt")

            if section:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(section)
                print(f"Extracted: {company}/{year} — {len(section.splitlines())} lines")
            else:
                print(f"NOT FOUND: {company}/{year} — check manually")