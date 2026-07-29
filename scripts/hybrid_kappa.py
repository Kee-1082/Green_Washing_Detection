"""
hybrid_kappa.py — Self-Agreement & LLM-Human Agreement Calculator
==================================================================
Two modes:

MODE 1: LLM-Human Agreement (run immediately after annotation)
  Computes Cohen's kappa between LLM suggested labels and Kee's labels
  across the annotated batch. Identifies override patterns.

MODE 2: Kee Self-Agreement Test (run 2 weeks later)
  - Selects 20 random sentences from the full annotated corpus
  - Generates a re-annotation batch (re_annotate_sample.csv)
  - After re-annotating, computes Week-1 vs Week-2 kappa

Usage:
  # LLM-Human agreement on a session
  python scripts/hybrid_kappa.py --mode llm-human --session 1

  # Generate 20-sentence self-agreement sample
  python scripts/hybrid_kappa.py --mode self-gen --seed 42

  # Compute self-agreement after re-annotating
  python scripts/hybrid_kappa.py --mode self-kappa --week1 kee_session1 --week2 kee_self
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import csv
import sys
import random
import argparse
from collections import Counter

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError:
    print("ERROR: scikit-learn not installed. Run: venv\\Scripts\\pip install scikit-learn")
    sys.exit(1)

ANNOTATIONS_DIR = "annotations"
LOG_PATH        = "logs/hybrid_kappa_log.txt"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def compute_kappa_report(labels_a, labels_b, name_a, name_b):
    """Compute and print kappa + confusion matrix."""
    # Filter out ? (unsure) from kappa calculation
    pairs = [(a, b) for a, b in zip(labels_a, labels_b)
             if a != "?" and b != "?"]
    if len(pairs) < 2:
        print("  Not enough non-? pairs to compute kappa.")
        return None, 0, 0

    la, lb = zip(*pairs)
    kappa  = cohen_kappa_score(la, lb)

    total   = len(labels_a)
    agreed  = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    pct     = agreed / total * 100

    if kappa >= 0.80:
        interp = "STRONG (>=0.80) ✓ — acceptable consistency"
    elif kappa >= 0.60:
        interp = "GOOD (0.60-0.80) — minor calibration recommended"
    elif kappa >= 0.40:
        interp = "MODERATE (0.40-0.60) — review disagreements"
    else:
        interp = "POOR (<0.40) — must revise schema / re-annotate"

    print("  Rater A        : {}".format(name_a))
    print("  Rater B        : {}".format(name_b))
    print("  Total compared : {} ({} after removing ?)".format(total, len(pairs)))
    print("  Exact agree    : {} ({:.1f}%)".format(agreed, pct))
    print("  Cohen's Kappa  : {:.4f}".format(kappa))
    print("  Interpretation : {}".format(interp))

    # Label distribution
    dist_a = Counter(labels_a)
    dist_b = Counter(labels_b)
    print("\n  Label Distribution:")
    print("    {:>8} | {:>8}  {:>8}".format("Label", name_a[:8], name_b[:8]))
    print("    " + "-" * 30)
    for lbl in sorted(set(list(dist_a.keys()) + list(dist_b.keys()))):
        print("    {:>8} | {:>8}  {:>8}".format(
            lbl, dist_a.get(lbl, 0), dist_b.get(lbl, 0)))

    # Confusion matrix
    all_labels = sorted(set(la + lb))
    cm = confusion_matrix(la, lb, labels=all_labels)
    print("\n  Confusion Matrix (rows={}, cols={}):".format(name_a, name_b))
    header = "         " + "  ".join("{:>5}".format(l) for l in all_labels)
    print(header)
    for j, lbl in enumerate(all_labels):
        row_str = "{:>9}".format(lbl)
        for val in cm[j]:
            row_str += "{:>7}".format(val)
        print(row_str)

    return kappa, agreed, total


def log_result(mode, msg):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        from datetime import datetime
        f.write("[{}] mode={} {}\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), mode, msg))


# ---------------------------------------------------------------------------
# MODE 1: LLM-Human Agreement
# ---------------------------------------------------------------------------

def mode_llm_human(session: int, rater: str):
    path = os.path.join(ANNOTATIONS_DIR,
                        "{}_session{}.csv".format(rater, session))
    if not os.path.exists(path):
        print("ERROR: {} not found.".format(path))
        sys.exit(1)

    rows = load_csv(path)
    if not rows:
        print("ERROR: No annotations found in {}.".format(path))
        sys.exit(1)

    llm_labels = [r["llm_label"] for r in rows]
    kee_labels = [r["kee_label"] for r in rows]

    print("\n" + "=" * 60)
    print("  LLM vs KEE AGREEMENT REPORT  (Session {})".format(session))
    print("=" * 60)

    kappa, agreed, total = compute_kappa_report(
        llm_labels, kee_labels, "LLM", "Kee")

    # Override analysis
    overrides = [r for r in rows
                 if r["llm_label"] != r["kee_label"] and r["kee_label"] != "?"]
    v_to_s = sum(1 for r in overrides
                 if r["llm_label"] == "V" and r["kee_label"] == "S")
    s_to_v = sum(1 for r in overrides
                 if r["llm_label"] == "S" and r["kee_label"] == "V")

    print("\n  Override Analysis:")
    print("    Total overrides        : {}".format(len(overrides)))
    print("    LLM said V → Kee said S: {} (LLM was too conservative)".format(v_to_s))
    print("    LLM said S → Kee said V: {} (LLM was too generous)".format(s_to_v))
    print("    Marked Unsure (?)      : {}".format(
        sum(1 for r in rows if r["kee_label"] == "?")))

    # Greenwashing risk distribution
    gw_high = [r for r in rows if float(r.get("llm_gw_risk", 0)) >= 0.65]
    print("\n  Greenwashing Risk (LLM-assessed):")
    print("    HIGH (>=0.65): {} / {}".format(len(gw_high), total))
    if gw_high:
        print("    HIGH-risk sentences:")
        for r in gw_high[:5]:
            print("      [{}] {}  kee={}".format(
                r["sentence_id"],
                r["sentence_text"][:60] + "...",
                r["kee_label"]))

    # Export override report
    if overrides:
        ov_path = os.path.join(ANNOTATIONS_DIR,
                               "overrides_session{}.csv".format(session))
        fields = ["sentence_id", "company", "year", "sentence_text",
                  "llm_label", "kee_label", "kee_confidence",
                  "kee_notes", "llm_gw_risk", "llm_top_features"]
        save_csv(ov_path, [{k: r.get(k, "") for k in fields}
                            for r in overrides], fields)
        print("\n  Overrides exported → {}".format(ov_path))

    if kappa is not None:
        log_result("llm-human",
                   "session={} kappa={:.4f} agree={}/{} overrides={}".format(
                       session, kappa, agreed, total, len(overrides)))

    print("=" * 60)


# ---------------------------------------------------------------------------
# MODE 2a: Generate Self-Agreement Sample
# ---------------------------------------------------------------------------

def mode_self_gen(seed: int, n: int, rater: str):
    # Collect all annotated sentences across sessions
    all_rows = []
    for fname in sorted(os.listdir(ANNOTATIONS_DIR)):
        if fname.startswith(rater + "_session") and fname.endswith(".csv"):
            fpath = os.path.join(ANNOTATIONS_DIR, fname)
            all_rows.extend(load_csv(fpath))

    if not all_rows:
        print("ERROR: No annotations found for rater '{}'. ".format(rater))
        print("  Annotate at least one session first.")
        sys.exit(1)

    random.seed(seed)
    sample = random.sample(all_rows, min(n, len(all_rows)))

    # Write re-annotation file (blank kee columns for re-labeling)
    sample_path = os.path.join(ANNOTATIONS_DIR, rater + "_self_sample.csv")
    fields = ["sentence_id", "company", "year", "report_type",
              "sentence_text", "llm_label", "llm_confidence",
              "llm_top_features", "llm_gw_risk",
              "kee_label", "kee_confidence", "kee_notes", "agreement_yn"]

    out_rows = []
    for r in sample:
        row = {k: r.get(k, "") for k in fields}
        # Clear Kee's old labels for re-annotation
        row["kee_label"]      = ""
        row["kee_confidence"] = ""
        row["kee_notes"]      = ""
        row["agreement_yn"]   = ""
        out_rows.append(row)

    save_csv(sample_path, out_rows, fields)

    print("\n" + "=" * 60)
    print("  SELF-AGREEMENT SAMPLE GENERATED")
    print("  Sentences selected : {} (seed={})".format(len(sample), seed))
    print("  File               : {}".format(sample_path))
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Wait ~2 weeks (to avoid memory effect)")
    print("  2. Re-annotate the sample file WITHOUT looking at your old labels")
    print("  3. Use: python scripts/xnlp_annotator.py --session self ...")
    print("  4. Then run: python scripts/hybrid_kappa.py --mode self-kappa")


# ---------------------------------------------------------------------------
# MODE 2b: Self-Kappa Computation
# ---------------------------------------------------------------------------

def mode_self_kappa(rater: str):
    # Week 1: original annotations merged across sessions
    week1_rows = {}
    for fname in sorted(os.listdir(ANNOTATIONS_DIR)):
        if fname.startswith(rater + "_session") and fname.endswith(".csv"):
            fpath = os.path.join(ANNOTATIONS_DIR, fname)
            for row in load_csv(fpath):
                week1_rows[row["sentence_id"]] = row["kee_label"]

    # Week 2: re-annotated sample
    sample_path = os.path.join(ANNOTATIONS_DIR, rater + "_self_sample.csv")
    if not os.path.exists(sample_path):
        print("ERROR: {} not found.".format(sample_path))
        sys.exit(1)

    week2_rows = {}
    for row in load_csv(sample_path):
        if row.get("kee_label"):  # only annotated rows
            week2_rows[row["sentence_id"]] = row["kee_label"]

    common = sorted(set(week1_rows.keys()) & set(week2_rows.keys()))
    if not common:
        print("ERROR: No common annotated sentences between week 1 and week 2.")
        sys.exit(1)

    w1_labels = [week1_rows[sid] for sid in common]
    w2_labels = [week2_rows[sid] for sid in common]

    print("\n" + "=" * 60)
    print("  SELF-AGREEMENT REPORT  (Kee Week 1 vs Week 2)")
    print("=" * 60)

    kappa, agreed, total = compute_kappa_report(
        w1_labels, w2_labels, "Kee-W1", "Kee-W2")

    if kappa is not None:
        log_result("self-kappa",
                   "n={} kappa={:.4f} agree={}/{}".format(
                       total, kappa, agreed, total))

        # Paper-ready sentence
        print("\n  ── PAPER-READY STATEMENT ──────────────────────────────")
        print("  \"Single annotator achieved κ={:.2f} on self-agreement".format(kappa))
        print("   test ({}-sentence sample, 2-week interval),".format(total))
        print("   validating annotation consistency.\"")
        print("  ──────────────────────────────────────────────────────")

    print("=" * 60)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Kappa Calculator (LLM-Human + Self-Agreement)")
    parser.add_argument("--mode",    required=True,
                        choices=["llm-human", "self-gen", "self-kappa"],
                        help="Operation mode")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number (for llm-human mode)")
    parser.add_argument("--rater",   default="kee",
                        help="Rater ID (default: kee)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for self-gen mode (default: 42)")
    parser.add_argument("--n",       type=int, default=20,
                        help="Sample size for self-gen mode (default: 20)")
    args = parser.parse_args()

    if args.mode == "llm-human":
        mode_llm_human(args.session, args.rater)
    elif args.mode == "self-gen":
        mode_self_gen(args.seed, args.n, args.rater)
    elif args.mode == "self-kappa":
        mode_self_kappa(args.rater)
