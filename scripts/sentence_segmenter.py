"""
Phase 2: Sentence Segmentation & Environmental Filtering
=========================================================
Reads extracted BRR/BRSR sections from data/brsr_sections/,
cleans multi-column PDF artifacts, segments into sentences via spaCy,
filters for environmental (E-pillar) sentences using keyword matching,
and outputs a structured CSV for annotation (Phase 3).

Pipeline
--------
1. TEXT CLEANING   - Remove page headers, table headers, bare numbers
2. SENTENCE SEG    - spaCy en_core_web_sm, 8-180 token bounds
3. ENV FILTER      - >= 1 primary keyword OR >= 2 secondary keywords
4. QUALITY FLAG    - table_row vs ok (for numeric-financial labeling)
5. DEDUP           - exact-match lowercase deduplication

Output: data/sentences/all_sentences.csv
        logs/phase2_stats.txt
"""

import os
import re
import csv
from datetime import datetime

import spacy

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_ROOT = "data/brsr_sections"
OUTPUT_CSV = "data/sentences/all_sentences.csv"
LOG_PATH   = "logs/phase2_stats.txt"

NLP = spacy.load("en_core_web_sm")
NLP.disable_pipes([p for p in ["ner", "lemmatizer"] if p in NLP.pipe_names])

PRIMARY_KEYWORDS = [
    "emission", "co2", "ghg", "greenhouse", "carbon",
    "scope 1", "scope 2", "scope 3", "decarboni",
    "net zero", "carbon neutral",
]

SECONDARY_KEYWORDS = [
    "renewable", "energy", "water", "waste", "effluent",
    "climate", "biodiversity", "pollution", "discharge",
    "solar", "wind", "recycl", "fossil", "fuel",
    "temperature", "ecology", "environmental",
]

MIN_TOKENS = 8
MAX_TOKENS = 180

# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------

DROP_PATTERNS = [
    re.compile(r"^\d{1,3}\s+[A-Z]{3,}", re.IGNORECASE),
    re.compile(r"^\s*[\d]+\s*$"),
    re.compile(r"^[A-Z][A-Z\s\d|]+$"),
]


def clean_text(raw):
    lines = raw.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if len(s) < 15:
            continue
        skip = any(p.match(s) for p in DROP_PATTERNS)
        if skip:
            continue
        alpha = [c for c in s if c.isalpha()]
        if alpha and len(s) < 90:
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            if upper_ratio > 0.80:
                continue
        kept.append(s)

    text = " ".join(kept)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ENVIRONMENTAL KEYWORD CHECK
# ---------------------------------------------------------------------------

def is_environmental(sl):
    ph = [kw for kw in PRIMARY_KEYWORDS if kw in sl]
    if ph:
        return True, ph, []
    sh = [kw for kw in SECONDARY_KEYWORDS if kw in sl]
    if len(sh) >= 2:
        return True, [], sh
    return False, [], []


# ---------------------------------------------------------------------------
# SENTENCE SEGMENTATION
# ---------------------------------------------------------------------------

def extract_env_sentences(text, company, year, report_type):
    results = []
    seen = set()

    try:
        doc = NLP(text)
    except Exception as e:
        print("  [spaCy error] " + str(e))
        return results

    for sent in doc.sents:
        s = sent.text.strip()

        tok_count = len([t for t in sent if not t.is_space])
        if tok_count < MIN_TOKENS or tok_count > MAX_TOKENS:
            continue

        # Quality guard: table-row detection
        alpha_chars = sum(1 for c in s if c.isalpha())
        alpha_ratio = alpha_chars / max(len(s), 1)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)
        quality = "table_row" if (alpha_ratio < 0.45 and not has_verb) else "ok"

        # Dedup
        key = re.sub(r"\s+", " ", s.lower().strip())
        if key in seen:
            continue
        seen.add(key)

        s_lower = s.lower()
        matched, ph, sh = is_environmental(s_lower)
        if not matched:
            continue

        results.append({
            "company":        company,
            "year":           year,
            "report_type":    report_type,
            "sentence_text":  s,
            "primary_hit":    "|".join(ph) if ph else "",
            "secondary_hits": "|".join(sh) if sh else "",
            "quality":        quality,
        })

    return results


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

os.makedirs("data/sentences", exist_ok=True)
os.makedirs("logs", exist_ok=True)

all_sentences = []
log_entries   = ["=== Phase 2 Sentence Segmentation: {} ===\n".format(
    datetime.now().isoformat())]
company_stats = {}

for company in sorted(os.listdir(INPUT_ROOT)):
    comp_path = os.path.join(INPUT_ROOT, company)
    if not os.path.isdir(comp_path):
        continue
    if company.lower() == "ultratect":
        print("[SKIP] ultratect/ (typo folder - use ultratech/)")
        continue

    for fname in sorted(os.listdir(comp_path)):
        if not fname.endswith(".txt"):
            continue

        year = fname.replace(".txt", "")
        try:
            yr = int(year)
        except ValueError:
            yr = 0
        report_type = "BRSR" if yr >= 2022 else "BRR"

        fpath = os.path.join(comp_path, fname)
        raw = open(fpath, encoding="utf-8").read()

        if raw.startswith("[EXTRACTION FAILED"):
            log_entries.append("[SKIP] company={} year={} - placeholder file".format(
                company, year))
            continue

        cleaned  = clean_text(raw)
        sentences = extract_env_sentences(cleaned, company, year, report_type)

        company_stats.setdefault(company, {"total": 0, "files": 0})
        company_stats[company]["total"] += len(sentences)
        company_stats[company]["files"] += 1
        all_sentences.extend(sentences)

        status = "[OK] " if sentences else "[WARN]"
        print("{} {}/{} ({}) - {} env sentences".format(
            status, company, year, report_type, len(sentences)))
        log_entries.append("{} company={} year={} type={} env_sentences={}".format(
            status.strip(), company, year, report_type, len(sentences)))

# Assign global sentence IDs
for i, row in enumerate(all_sentences):
    row["sentence_id"] = "S{:04d}".format(i + 1)

FIELDNAMES = ["sentence_id", "company", "year", "report_type",
              "sentence_text", "primary_hit", "secondary_hits", "quality"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    for row in all_sentences:
        writer.writerow({k: row[k] for k in FIELDNAMES})

total = len(all_sentences)
log_entries.append("\n--- Company Stats ---")
for comp, stats in company_stats.items():
    log_entries.append("  {}: {} sentences across {} files".format(
        comp, stats["total"], stats["files"]))
log_entries.append("\nTOTAL CORPUS: {} environmental sentences".format(total))
log_entries.append("Output: {}".format(OUTPUT_CSV))

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_entries))

print("\n" + "=" * 60)
print("CORPUS TOTAL: {} environmental sentences".format(total))
print("Output CSV  : " + OUTPUT_CSV)
print("Stats log   : " + LOG_PATH)
