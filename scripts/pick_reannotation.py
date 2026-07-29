"""
Pick Re-annotation Candidates
==============================
After the annotation re-alignment, 33 of 50 manual labels were lost
because their source sentences were bleed artifacts that no longer exist
in the clean corpus.

This script selects the best 33 replacement sentences from the new corpus
for fast re-annotation, prioritising:
  1. Sentences from the SAME company/year as the lost annotations
  2. Sentences not already labeled (not in the 17 recovered labels)
  3. Balanced across label-ambiguous types (vague, quantitative, descriptive)

Output:
  annotations/reannotation_batch.csv   — 33 sentences for manual labeling
  (formatted identically to kee_session1.csv for easy merge)
"""

import os
import csv
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from feature_extractor import suggest_label

ANNOTATIONS_DIR = "annotations"
NEW_CORPUS      = "data/sentences/all_sentences.csv"
OUT_PATH        = os.path.join(ANNOTATIONS_DIR, "reannotation_batch.csv")

# Companies/years that had the most unmatched (lost) annotations
LOST_ORIGINS = [
    ("jsw", "2019"), ("jsw", "2020"),  # sessions 1 & 2 were mostly jsw 2019-20
]

# IDs already locked in (17 recovered labels)
REALIGNED_IDS = set()
for fname in ["kee_session1_realigned.csv", "kee_session2_realigned.csv"]:
    fpath = os.path.join(ANNOTATIONS_DIR, fname)
    if os.path.exists(fpath):
        for row in csv.DictReader(open(fpath, encoding="utf-8")):
            sid = row["sentence_id"]
            if not sid.startswith("UNMATCHED"):
                REALIGNED_IDS.add(sid)


def score_annotation_value(text, llm_label, llm_conf, gw_risk):
    """
    Score how valuable this sentence is to annotate.
    We want ambiguous sentences (not trivially N) that have ESG content.
    """
    # Prefer V and S candidates (more informative for greenwash detection)
    label_score = {"V": 3, "S": 2, "N": 1}.get(llm_label, 1)
    # Prefer lower LLM confidence (more need for human judgment)
    conf_score = 1 - llm_conf
    # Prefer higher GW risk
    gw_score = gw_risk
    return label_score + conf_score + gw_score


def run():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    # Load new corpus
    all_rows = list(csv.DictReader(open(NEW_CORPUS, encoding="utf-8")))
    print(f"Loaded {len(all_rows)} sentences from new corpus")
    print(f"Already labeled (recovered): {len(REALIGNED_IDS)} sentences")

    # Filter: not already labeled
    candidates = [r for r in all_rows if r["sentence_id"] not in REALIGNED_IDS]

    # Priority 1: same company/year as lost annotations
    priority   = [r for r in candidates
                  if (r["company"], r["year"]) in LOST_ORIGINS]
    remainder  = [r for r in candidates
                  if (r["company"], r["year"]) not in LOST_ORIGINS]

    print(f"Priority candidates (jsw 2019-20): {len(priority)}")
    print(f"Remaining pool: {len(remainder)}")

    # Score + rank each candidate
    scored = []
    for row in (priority + remainder):
        text = row["sentence_text"]
        try:
            llm_label, llm_conf, _, gw_risk, _ = suggest_label(
                text, row.get("quality", "ok"))
        except Exception:
            llm_label, llm_conf, gw_risk = "N", 0.5, 0.0

        val = score_annotation_value(text, llm_label, llm_conf, gw_risk)
        scored.append((val, row, llm_label, llm_conf, gw_risk))

    # Sort by value descending, pick top 33
    scored.sort(key=lambda x: -x[0])
    selected = scored[:33]

    print(f"\nSelected {len(selected)} sentences for re-annotation")

    # Write output CSV in kee_session format
    fields = [
        "sentence_id", "company", "year", "report_type", "sentence_text",
        "llm_label", "llm_gw_risk", "kee_label", "kee_confidence", "kee_notes",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for val, row, llm_label, llm_conf, gw_risk in selected:
            w.writerow({
                "sentence_id":    row["sentence_id"],
                "company":        row["company"],
                "year":           row["year"],
                "report_type":    row["report_type"],
                "sentence_text":  row["sentence_text"],
                "llm_label":      llm_label,
                "llm_gw_risk":    f"{gw_risk:.2f}",
                "kee_label":      "",   # to be filled by annotator
                "kee_confidence": "",
                "kee_notes":      "",
            })

    print(f"Written: {OUT_PATH}")
    print("\nNext step: manually fill in 'kee_label' column in reannotation_batch.csv")
    print("Then run:  python scripts/train_predict.py")
    print("           (after updating it to also load reannotation_batch.csv)")

    # Preview top 10 candidates
    print("\n=== Top 10 candidates for re-annotation ===")
    for i, (val, row, llm_label, llm_conf, gw_risk) in enumerate(selected[:10], 1):
        text_preview = row["sentence_text"][:120].replace("\n", " ")
        print(f"  {i:2d}. [{row['sentence_id']}] {row['company']}/{row['year']} "
              f"llm={llm_label}(conf={llm_conf:.2f}) gw={gw_risk:.2f}")
        print(f"      {text_preview}")


if __name__ == "__main__":
    run()
