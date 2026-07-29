"""
Annotation Re-alignment
========================
After re-segmenting the corpus with clean text, sentence IDs no longer
match the manual annotations in kee_session*.csv.

This script re-aligns annotations by matching annotated sentence TEXT
to the new corpus sentences using:
  1. Exact text match
  2. First-80-char prefix match (handles minor whitespace differences)
  3. Token-overlap ratio (fuzzy match — catches truncation artifacts)

Outputs:
  annotations/kee_session1_realigned.csv   — annotations with updated IDs
  annotations/kee_session2_realigned.csv   — same
  logs/realignment_report.txt              — match summary + unmatched list
"""

import os
import re
import csv
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANNOTATIONS_DIR = "annotations"
NEW_CORPUS      = "data/sentences/all_sentences.csv"
LOG_PATH        = "logs/realignment_report.txt"


def normalise(text):
    """Normalise text for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def token_overlap(a, b):
    """Jaccard token overlap ratio between two strings."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_new_index(new_corpus_path):
    """Build lookup structures from the new corpus."""
    rows = list(csv.DictReader(open(new_corpus_path, encoding="utf-8")))

    # exact normalised text → sentence_id
    exact_idx = {}
    # prefix (first 80 chars normalised) → list of (sentence_id, full_text)
    prefix_idx = {}
    # full list for fuzzy fallback
    all_rows = []

    for row in rows:
        norm = normalise(row["sentence_text"])
        exact_idx[norm] = row["sentence_id"]
        prefix = norm[:80]
        prefix_idx.setdefault(prefix, []).append((row["sentence_id"], norm))
        all_rows.append((row["sentence_id"], norm))

    return exact_idx, prefix_idx, all_rows, rows


def find_match(old_text, exact_idx, prefix_idx, all_rows, fuzzy_threshold=0.55):
    """
    Try to find the new sentence_id that matches this old text.
    Returns (new_id, match_type) or (None, 'UNMATCHED').
    """
    norm = normalise(old_text)

    # 1. Exact match
    if norm in exact_idx:
        return exact_idx[norm], "exact"

    # 2. Prefix match (first 80 chars)
    prefix = norm[:80]
    if prefix in prefix_idx:
        candidates = prefix_idx[prefix]
        if len(candidates) == 1:
            return candidates[0][0], "prefix"
        # Multiple prefix matches: pick best token overlap
        best_id, best_score = None, 0
        for sid, cand_norm in candidates:
            sc = token_overlap(norm, cand_norm)
            if sc > best_score:
                best_score = sc
                best_id = sid
        if best_score >= fuzzy_threshold:
            return best_id, f"prefix_fuzzy({best_score:.2f})"

    # 3. Fuzzy fallback: scan all rows (expensive but only ~50 annotations)
    best_id, best_score = None, 0
    for sid, cand_norm in all_rows:
        sc = token_overlap(norm, cand_norm)
        if sc > best_score:
            best_score = sc
            best_id = sid

    if best_score >= fuzzy_threshold:
        return best_id, f"fuzzy({best_score:.2f})"

    return None, "UNMATCHED"


def realign_session(fname, exact_idx, prefix_idx, all_rows, new_rows_by_id):
    """
    Re-align a single kee_session CSV.
    Returns (realigned_rows, report_lines).
    """
    fpath = os.path.join(ANNOTATIONS_DIR, fname)
    rows  = list(csv.DictReader(open(fpath, encoding="utf-8")))

    out_rows     = []
    report_lines = [f"\n=== {fname} ==="]
    matched      = 0
    unmatched    = []

    for row in rows:
        old_id   = row["sentence_id"]
        old_text = row.get("sentence_text", "")

        new_id, match_type = find_match(old_text, exact_idx, prefix_idx, all_rows)

        if new_id:
            # Update the sentence_id and sentence_text to the new corpus values
            new_row = dict(row)
            new_row["sentence_id"]   = new_id
            new_row["sentence_text"] = new_rows_by_id.get(new_id, {}).get(
                "sentence_text", old_text)
            out_rows.append(new_row)
            matched += 1
            report_lines.append(
                f"  [OK]  {old_id} -> {new_id}  [{match_type}]  "
                f"{old_text[:60].replace(chr(10),' ')}")
        else:
            # Keep the old row but flag it — can't find it in new corpus
            new_row = dict(row)
            new_row["sentence_id"] = f"UNMATCHED_{old_id}"
            out_rows.append(new_row)
            unmatched.append(old_id)
            report_lines.append(
                f"  [MISS] {old_id} -> UNMATCHED  "
                f"{old_text[:60].replace(chr(10),' ')}")

    report_lines.append(
        f"\n  Result: {matched}/{len(rows)} matched, "
        f"{len(unmatched)} unmatched")
    if unmatched:
        report_lines.append(f"  Unmatched IDs: {', '.join(unmatched)}")

    return out_rows, report_lines, matched, len(unmatched)


def run():
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("ANNOTATION RE-ALIGNMENT")
    print("=" * 60)

    # Build new corpus index
    exact_idx, prefix_idx, all_rows, new_rows = build_new_index(NEW_CORPUS)
    new_rows_by_id = {r["sentence_id"]: r for r in new_rows}
    print(f"\nNew corpus loaded: {len(new_rows)} sentences")

    # Find session files
    session_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.startswith("kee_session") and f.endswith(".csv")
        and "realigned" not in f
    ])
    print(f"Session files found: {session_files}")

    all_report = ["ANNOTATION RE-ALIGNMENT REPORT", "=" * 60]
    total_matched   = 0
    total_unmatched = 0

    for fname in session_files:
        out_rows, report_lines, matched, unmatched_n = realign_session(
            fname, exact_idx, prefix_idx, all_rows, new_rows_by_id)

        # Write realigned CSV with same columns + updated IDs
        out_fname = fname.replace(".csv", "_realigned.csv")
        out_path  = os.path.join(ANNOTATIONS_DIR, out_fname)

        # Preserve original fieldnames
        orig_fields = list(csv.DictReader(
            open(os.path.join(ANNOTATIONS_DIR, fname),
                 encoding="utf-8")).fieldnames)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=orig_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)

        all_report.extend(report_lines)
        total_matched   += matched
        total_unmatched += unmatched_n

        print(f"\n  {fname}: {matched} matched, {unmatched_n} unmatched")
        print(f"  Written: {out_path}")

    # Summary
    all_report.append(f"\n{'='*60}")
    all_report.append(f"TOTAL: {total_matched} matched, {total_unmatched} unmatched")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(all_report))

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_matched} matched, {total_unmatched} unmatched")
    print(f"Report: {LOG_PATH}")

    if total_unmatched == 0:
        print("\nAll annotations re-aligned. Ready to retrain.")
        print("Run: python scripts/train_predict.py")
        print("     (update INPUT annotations to use *_realigned.csv)")
    else:
        print(f"\n{total_unmatched} annotations could not be matched.")
        print("Check logs/realignment_report.txt for details.")
        print("These may need manual re-annotation.")


if __name__ == "__main__":
    run()
