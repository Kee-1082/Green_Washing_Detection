"""
train_predict.py  — Phase 4: Train on 50 Manual Labels + Predict Remaining Corpus
==================================================================================
Workflow:
  1. Load 50 manually annotated sentences from annotations/kee_session*.csv
  2. Extract 5 linguistic features per sentence (from feature_extractor.py)
  3. Train a Multinomial Naive Bayes classifier (explainable, no black-box)
  4. Predict labels for all remaining ~1,122 unlabeled sentences
  5. Flag predictions with confidence < CONF_THRESHOLD for manual QA review
  6. Export two CSVs:
       - annotations/full_corpus_labeled.csv  (all 1,172 sentences)
       - annotations/low_confidence_review.csv (subset for manual QA)

Explainability Design:
  - Features are interpretable numeric scores (0-1), not embeddings
  - Naive Bayes is inherently probabilistic → gives class probabilities
  - Low-confidence = max class probability < CONF_THRESHOLD (default 0.65)
  - Model decision + confidence stored in full_corpus_labeled.csv

Usage:
  # After completing 50-sentence manual annotation:
  python scripts/train_predict.py

  # Review the flagged low-confidence predictions:
  python scripts/train_predict.py --review-only

  # Custom confidence threshold (default: 0.65):
  python scripts/train_predict.py --threshold 0.70

Paper documentation:
  "Hybrid annotation: 50 sentences manually labeled using LLM-assisted feature
  salience (xNLP framework); remaining 1,122 sentences predicted by a Multinomial
  Naive Bayes classifier trained on 5 interpretable linguistic features. Manual QA
  applied to all low-confidence predictions (p < 0.65), yielding a fully labeled
  corpus of 1,172 sentences."
"""

import os
import csv
import sys
import argparse
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report
except ImportError:
    print("ERROR: scikit-learn not installed.")
    print("Run: venv\\Scripts\\pip install scikit-learn")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import (
    feat_vague_adj_ratio, feat_quantifier_count,
    feat_verb_strength, feat_target_year, feat_specific_tech,
    suggest_label,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_CSV          = "data/sentences/all_sentences.csv"
ANNOTATIONS_DIR    = "annotations"
FULL_LABELED_CSV   = "annotations/full_corpus_labeled.csv"
LOW_CONF_CSV       = "annotations/low_confidence_review.csv"
LOG_PATH           = "logs/train_predict_log.txt"

CONF_THRESHOLD     = 0.65   # below this → flag for manual QA
LABEL_MAP          = {"V": 0, "S": 1, "N": 2}
LABEL_INV          = {0: "V", 1: "S", 2: "N"}

FULL_FIELDS = [
    "sentence_id", "company", "year", "report_type", "sentence_text",
    "source",          # "manual" | "model" | "model_reviewed"
    "label",           # final agreed label
    "confidence",      # model probability (or "1.0" for manual)
    "llm_label",       # rule-based LLM suggestion
    "llm_gw_risk",     # greenwashing risk score
    "llm_top_features",# feature CSV string
    "kee_label",       # manual label (blank for model-predicted rows)
    "kee_confidence",  # annotator confidence
    "kee_notes",       # annotator notes
    "flagged_for_qa",  # "Y" if low confidence model prediction
]


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_features(text: str, quality: str = "ok") -> list:
    """Extract 5 numeric features → [vague, quant, verb, year, tech]."""
    text_lower = text.lower()
    import re
    tokens = re.findall(r"\b[a-z]+\b", text_lower)

    f_vague = feat_vague_adj_ratio(text_lower, tokens)
    f_quant = feat_quantifier_count(text_lower)
    f_verb  = feat_verb_strength(text_lower)
    f_year  = feat_target_year(text_lower)
    f_tech  = feat_specific_tech(text_lower)

    return [f_vague.score, f_quant.score, f_verb.score,
            f_year.score, f_tech.score]

FEATURE_NAMES = [
    "vague_adj_ratio", "quantifier_count", "verb_strength",
    "target_year", "specific_tech"
]


# ---------------------------------------------------------------------------
# LOAD MANUAL ANNOTATIONS
# ---------------------------------------------------------------------------

def load_manual_annotations():
    """
    Merge all annotation session files → dict of sentence_id → row.

    Loads in priority order:
      1. kee_session*_realigned.csv  — original annotations re-aligned to new corpus
      2. reannotation_batch.csv      — fresh annotations on new clean sentences
      3. kee_session*.csv            — original files (fallback if no realigned exists)
    """
    manual = {}

    # Prefer realigned versions of original sessions
    realigned_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.startswith("kee_session") and f.endswith("_realigned.csv")
    ])
    # Re-annotation batch for the 33 replaced sentences
    batch_files = ["reannotation_batch.csv"]
    # Fallback: original sessions if no realigned exists
    orig_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.startswith("kee_session") and f.endswith(".csv")
        and "_realigned" not in f
    ])

    # Use realigned if present, else fall back to originals
    session_files = (realigned_files if realigned_files else orig_files) + [
        f for f in batch_files
        if os.path.exists(os.path.join(ANNOTATIONS_DIR, f))
    ]

    if not session_files:
        print("ERROR: No annotation session files found in " + ANNOTATIONS_DIR)
        print("Complete annotation with: python scripts/xnlp_annotator.py ...")
        sys.exit(1)

    for fname in session_files:
        fpath = os.path.join(ANNOTATIONS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        loaded_before = len(manual)
        with open(fpath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["sentence_id"]
                # Skip unmatched placeholders and unsure labels
                if sid.startswith("UNMATCHED"):
                    continue
                if row.get("kee_label", "") in ("V", "S", "N"):
                    manual[sid] = row
        added = len(manual) - loaded_before
        print("[LOADED] {} -> {} new labeled sentences (total: {})".format(
            fname, added, len(manual)))

    # Also load model_reviewed rows from full_corpus_labeled.csv
    # These are QA-reviewed predictions the analyst has confirmed/corrected.
    full_labeled_path = FULL_LABELED_CSV
    if os.path.exists(full_labeled_path):
        loaded_before = len(manual)
        with open(full_labeled_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["sentence_id"]
                if row.get("source", "") == "model_reviewed" and \
                   row.get("kee_label", "") in ("V", "S", "N") and \
                   sid not in manual:
                    manual[sid] = row
        added = len(manual) - loaded_before
        if added > 0:
            print("[LOADED] full_corpus_labeled.csv (model_reviewed) -> {} new labeled sentences (total: {})".format(
                added, len(manual)))

    return manual



# ---------------------------------------------------------------------------
# TRAIN NAIVE BAYES
# ---------------------------------------------------------------------------

def train_model(manual_dict, all_rows):
    """Train MultinomialNB on the manually labeled sentences."""
    # Build training set
    X_train, y_train, train_ids = [], [], []
    for row in all_rows:
        sid = row["sentence_id"]
        if sid not in manual_dict:
            continue
        kee_label = manual_dict[sid].get("kee_label", "")
        if kee_label not in LABEL_MAP:
            continue
        feats = extract_features(row["sentence_text"], row.get("quality", "ok"))
        X_train.append(feats)
        y_train.append(LABEL_MAP[kee_label])
        train_ids.append(sid)

    if len(X_train) < 5:
        print("ERROR: Need at least 5 labeled sentences to train. Found: " + str(len(X_train)))
        sys.exit(1)

    # Scale to [0, 1] — MultinomialNB requires non-negative features
    # Features are already in [0,1] but MinMaxScaler ensures no negatives from rounding
    scaler = MinMaxScaler()
    X_np = np.array(X_train)
    X_scaled = scaler.fit_transform(X_np)
    # Add small epsilon to avoid zero (MultinomialNB needs strictly positive)
    X_scaled = X_scaled + 1e-6

    y_np = np.array(y_train)
    dist = Counter(y_train)
    print("\n[TRAINING] Naive Bayes on {} manually labeled sentences".format(len(X_train)))
    print("  Label distribution: " + str({LABEL_INV[k]: v for k, v in dist.items()}))

    # Check class balance warning
    total = len(y_train)
    for lbl, cnt in dist.items():
        pct = cnt / total * 100
        if pct < 10:
            print("  WARNING: Label '{}' has only {}% of training data ({} samples)".format(
                LABEL_INV[lbl], round(pct, 1), cnt))
            print("           Consider annotating more '{}' examples.".format(LABEL_INV[lbl]))

    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    model.fit(X_scaled, y_np)

    # In-sample accuracy (training set only — overfitting check)
    train_preds = model.predict(X_scaled)
    train_acc = (train_preds == y_np).mean()
    print("  In-sample accuracy: {:.1f}% (on training set)".format(train_acc * 100))
    print("  Note: True generalization measured on held-out low-confidence QA")

    return model, scaler, train_ids


# ---------------------------------------------------------------------------
# PREDICT + EXPORT
# ---------------------------------------------------------------------------

def predict_and_export(model, scaler, manual_dict, all_rows, threshold):
    """Predict labels for all non-manual sentences and export full corpus."""
    full_labeled  = []
    low_conf_rows = []
    stats = {"manual": 0, "model_high": 0, "model_low": 0}

    for row in all_rows:
        sid  = row["sentence_id"]
        text = row["sentence_text"]
        qual = row.get("quality", "ok")

        # Get LLM rule-based suggestion for all rows
        llm_label, llm_conf, top3, gw_risk, _ = suggest_label(text, qual)
        from feature_extractor import format_features_for_csv
        feat_csv = format_features_for_csv(top3)

        if sid in manual_dict:
            # ── Manual row ──────────────────────────────────────────────────
            ann = manual_dict[sid]
            out_row = {
                "sentence_id":      sid,
                "company":          row["company"],
                "year":             row["year"],
                "report_type":      row["report_type"],
                "sentence_text":    text,
                "source":           "manual",
                "label":            ann["kee_label"],
                "confidence":       "1.00",
                "llm_label":        ann.get("llm_label", llm_label),
                "llm_gw_risk":      ann.get("llm_gw_risk", str(gw_risk)),
                "llm_top_features": ann.get("llm_top_features", feat_csv),
                "kee_label":        ann["kee_label"],
                "kee_confidence":   ann.get("kee_confidence", "1"),
                "kee_notes":        ann.get("kee_notes", ""),
                "flagged_for_qa":   "N",
            }
            full_labeled.append(out_row)
            stats["manual"] += 1

        else:
            # ── Model prediction row ─────────────────────────────────────────
            feats  = extract_features(text, qual)
            X      = np.array([feats])
            X_sc   = scaler.transform(X) + 1e-6
            proba  = model.predict_proba(X_sc)[0]
            pred_idx   = int(np.argmax(proba))
            pred_label = LABEL_INV[pred_idx]
            pred_conf  = float(proba[pred_idx])
            flagged    = "Y" if pred_conf < threshold else "N"

            out_row = {
                "sentence_id":      sid,
                "company":          row["company"],
                "year":             row["year"],
                "report_type":      row["report_type"],
                "sentence_text":    text,
                "source":           "model",
                "label":            pred_label,
                "confidence":       "{:.4f}".format(pred_conf),
                "llm_label":        llm_label,
                "llm_gw_risk":      str(gw_risk),
                "llm_top_features": feat_csv,
                "kee_label":        "",
                "kee_confidence":   "",
                "kee_notes":        "",
                "flagged_for_qa":   flagged,
            }
            full_labeled.append(out_row)

            if flagged == "Y":
                low_conf_rows.append(out_row)
                stats["model_low"] += 1
            else:
                stats["model_high"] += 1

    # ── Write full corpus CSV ────────────────────────────────────────────────
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    with open(FULL_LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FULL_FIELDS)
        w.writeheader()
        w.writerows(full_labeled)

    # ── Write low-confidence review CSV ─────────────────────────────────────
    review_fields = ["sentence_id", "company", "year", "report_type",
                     "sentence_text", "label", "confidence",
                     "llm_label", "llm_gw_risk",
                     "llm_top_features", "kee_label", "kee_notes"]
    with open(LOW_CONF_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=review_fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in review_fields}
                     for r in low_conf_rows])

    return full_labeled, low_conf_rows, stats


# ---------------------------------------------------------------------------
# REVIEW MODE: update low-confidence labels after manual QA
# ---------------------------------------------------------------------------

def review_low_confidence():
    """
    Interactive review of low-confidence model predictions.
    Load LOW_CONF_CSV → show each → accept/override → update full corpus.
    """
    if not os.path.exists(LOW_CONF_CSV):
        print("ERROR: " + LOW_CONF_CSV + " not found. Run train_predict.py first.")
        sys.exit(1)
    if not os.path.exists(FULL_LABELED_CSV):
        print("ERROR: " + FULL_LABELED_CSV + " not found. Run train_predict.py first.")
        sys.exit(1)

    # Load both files
    low_conf = list(csv.DictReader(open(LOW_CONF_CSV, encoding="utf-8")))
    full     = list(csv.DictReader(open(FULL_LABELED_CSV, encoding="utf-8")))
    full_idx = {r["sentence_id"]: i for i, r in enumerate(full)}

    reviewed = 0
    print("\n" + "=" * 70)
    print("  LOW-CONFIDENCE REVIEW  ({} sentences flagged)".format(len(low_conf)))
    print("  Model confidence was below threshold for these predictions.")
    print("  Keys: Y=accept model | N=override | ?=unsure | Q=quit")
    print("=" * 70)

    for r in low_conf:
        sid     = r["sentence_id"]
        model_l = r["label"]
        conf    = r["confidence"]

        print("\n[{}] {}/{}  conf={} (model={})".format(
            sid, r["company"], r["year"], conf, model_l))
        print("-" * 70)
        text = r["sentence_text"]
        print(text[:280] + ("..." if len(text) > 280 else ""))
        print("  LLM suggestion : [{}]  GW risk: {}".format(
            r.get("llm_label", "?"), r.get("llm_gw_risk", "?")))
        print("-" * 70)

        while True:
            raw = input("  Accept model [{}]? [Y/N/?/Q]: ".format(model_l)).strip().upper()
            if raw == "Q":
                break
            elif raw == "Y":
                # Mark as reviewed, keep model label
                if sid in full_idx:
                    full[full_idx[sid]]["source"] = "model_reviewed"
                    full[full_idx[sid]]["flagged_for_qa"] = "N"
                reviewed += 1
                break
            elif raw == "N":
                while True:
                    new_l = input("  Your label [V/S/N]: ").strip().upper()
                    if new_l in ("V", "S", "N"):
                        break
                    print("  [Use V S N]")
                note = input("  Note: ").strip()
                if sid in full_idx:
                    full[full_idx[sid]]["label"]       = new_l
                    full[full_idx[sid]]["source"]      = "model_reviewed"
                    full[full_idx[sid]]["kee_label"]   = new_l
                    full[full_idx[sid]]["kee_notes"]   = note
                    full[full_idx[sid]]["flagged_for_qa"] = "N"
                reviewed += 1
                print("  [Updated: {} -> {}]".format(model_l, new_l))
                break
            elif raw == "?":
                if sid in full_idx:
                    full[full_idx[sid]]["kee_label"]   = "?"
                    full[full_idx[sid]]["source"]      = "model_reviewed"
                    full[full_idx[sid]]["flagged_for_qa"] = "N"
                reviewed += 1
                break
            else:
                print("  [Use Y N ? Q]")

        if raw == "Q":
            break

    # Save updated full corpus
    with open(FULL_LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FULL_FIELDS)
        w.writeheader()
        w.writerows(full)

    print("\n  Reviewed: {}/{} low-confidence sentences".format(reviewed, len(low_conf)))
    print("  Full corpus updated: " + FULL_LABELED_CSV)


# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------

def print_summary(full_labeled, low_conf_rows, stats, threshold):
    total = len(full_labeled)
    label_dist = Counter(r["label"] for r in full_labeled)
    gw_high = sum(1 for r in full_labeled
                  if float(r.get("llm_gw_risk", 0)) >= 0.65)
    model_rows = [r for r in full_labeled if r["source"].startswith("model")]
    avg_conf_model = (
        sum(float(r["confidence"]) for r in model_rows) / len(model_rows)
        if model_rows else 0
    )

    print("\n" + "=" * 70)
    print("  TRAIN + PREDICT COMPLETE")
    print("=" * 70)
    print("  Total corpus labeled : {}".format(total))
    print("  Manual labels        : {}".format(stats["manual"]))
    print("  Model labels (high)  : {}  (conf >= {})".format(
        stats["model_high"], threshold))
    print("  Model labels (low*)  : {}  (conf <  {} → QA needed)".format(
        stats["model_low"], threshold))
    print()
    print("  Final label distribution:")
    for lbl in ["V", "S", "N"]:
        cnt = label_dist.get(lbl, 0)
        pct = cnt / total * 100 if total else 0
        bar = "#" * int(pct / 2)
        names = {"V": "Vague", "S": "Substantive", "N": "Numeric-fin"}
        print("    [{}] {:15s} {:4d}  ({:5.1f}%)  {}".format(
            lbl, names[lbl], cnt, pct, bar))
    print()
    print("  Avg model confidence : {:.3f}".format(avg_conf_model))
    print("  GW HIGH risk count   : {} sentences".format(gw_high))
    print()
    print("  Full labeled corpus  : " + FULL_LABELED_CSV)
    print("  Low-conf for QA      : " + LOW_CONF_CSV +
          " ({} sentences)".format(len(low_conf_rows)))
    print("=" * 70)

    if low_conf_rows:
        print("\n  NEXT STEP: Review low-confidence predictions:")
        print("  python scripts/train_predict.py --review-only")

    print("\n  PAPER STATEMENT:")
    print("  'Hybrid annotation: {} sentences manually labeled (solo annotator,".format(
        stats["manual"]))
    print("   LLM-assisted feature salience); {} sentences predicted by a Multinomial".format(
        len(model_rows)))
    print("   Naive Bayes classifier trained on 5 interpretable linguistic features")
    print("   (vague_adj_ratio, quantifier_count, verb_strength, target_year,")
    print("   specific_tech). Manual QA applied to {} low-confidence predictions".format(
        stats["model_low"]))
    print("   (p < {}), yielding a fully labeled corpus of {} sentences.'".format(
        threshold, total))


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def write_log(stats, threshold, total, low_conf_n):
    os.makedirs("logs", exist_ok=True)
    from datetime import datetime
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("[{}] manual={} model_high={} model_low={} threshold={} total={}\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            stats["manual"], stats["model_high"], stats["model_low"],
            threshold, total))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NB on 50 manual labels, predict remaining corpus")
    parser.add_argument("--threshold",   type=float, default=CONF_THRESHOLD,
                        help="Confidence threshold for QA flag (default: 0.65)")
    parser.add_argument("--review-only", action="store_true",
                        help="Only run the low-confidence review step")
    args = parser.parse_args()

    if args.review_only:
        review_low_confidence()
        sys.exit(0)

    if not os.path.exists(INPUT_CSV):
        print("ERROR: " + INPUT_CSV + " not found. Run sentence_segmenter.py first.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  PHASE 4: TRAIN (50 manual) + PREDICT (rest)")
    print("  Confidence threshold for QA flag: " + str(args.threshold))
    print("=" * 70)

    # Step 1: load corpus
    all_rows = list(csv.DictReader(open(INPUT_CSV, encoding="utf-8")))
    print("\n[1/4] Corpus loaded: {} sentences".format(len(all_rows)))

    # Step 2: load manual annotations
    print("\n[2/4] Loading manual annotations...")
    manual_dict = load_manual_annotations()
    print("      Total manual labels loaded: {}".format(len(manual_dict)))

    # Step 3: train
    print("\n[3/4] Training Naive Bayes classifier...")
    model, scaler, train_ids = train_model(manual_dict, all_rows)

    # Step 4: predict + export
    print("\n[4/4] Predicting remaining {} sentences...".format(
        len(all_rows) - len(manual_dict)))
    full_labeled, low_conf_rows, stats = predict_and_export(
        model, scaler, manual_dict, all_rows, args.threshold)

    write_log(stats, args.threshold, len(full_labeled), len(low_conf_rows))
    print_summary(full_labeled, low_conf_rows, stats, args.threshold)
