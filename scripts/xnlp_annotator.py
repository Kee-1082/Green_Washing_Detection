"""
xnlp_annotator.py — Hybrid Annotation Tool (LLM-Assisted Solo Annotation)
===========================================================================
Implements the 2-tier annotation workflow:
  TIER 1: Rule-based LLM suggests label + confidence
  TIER 2: Top 3 linguistic features + greenwashing risk displayed
  KEE validates or overrides in 30-45 seconds

Output CSV schema:
  sentence_id, company, year, report_type, sentence_text,
  llm_label, llm_confidence, llm_top_features, llm_gw_risk,
  kee_label, kee_confidence, kee_notes, agreement_yn

Usage:
  # Start annotating Session 1 (sentences 1-25)
  python scripts/xnlp_annotator.py --session 1 --start 1 --end 25

  # Session 2 (sentences 26-50)
  python scripts/xnlp_annotator.py --session 2 --start 26 --end 50 --resume

  # Then train + predict remaining ~1,122 sentences:
  python scripts/train_predict.py

  # Review low-confidence model predictions:
  python scripts/train_predict.py --review-only

Keys during annotation:
  Y      Accept LLM label (then enter your confidence 1/2/3)
  N      Override (then enter your label + confidence)
  ?      Mark as Unsure (override to ?)
  H      Show schema reference
  B      Undo last annotation
  Q      Save and quit
  X      Express: accept THIS sentence with LLM confidence (no keypress)
"""

import os
import csv
import sys
import re
import argparse
from datetime import datetime

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError with box chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import our feature extractor
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import suggest_label, format_features_for_csv, display_analysis

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_CSV  = "data/sentences/all_sentences.csv"
OUTPUT_DIR = "annotations"
LOG_PATH   = "logs/xnlp_annotation_log.txt"

SCHEMA_QUICK = """
+======================================================================+
|  ANNOTATION SCHEMA (QUICK REF)                                       |
+======================================================================+
|  V  VAGUE          No numbers/dates/targets. Hedge verbs.            |
|     "committed to", "aim to", "promote", "strive" (without data)     |
|                                                                      |
|  S  SUBSTANTIVE    Has numbers, %, years, tech names, locations.     |
|     "reduced 15% by FY30", "50 MW solar commissioned FY23"          |
|                                                                      |
|  N  NUMERIC-FIN    Data row. No full sentence. Mostly units/numbers. |
|     "1,23,456 tCO2e | 3.8 m3/tcs (FY24) vs 4.1 (FY23)"            |
|                                                                      |
|  ?  UNSURE         Mixed signals. Flag for later review.            |
+----------------------------------------------------------------------+
|  KEYS:  Y=accept LLM | N=override | ?=unsure | B=back | Q=quit      |
+======================================================================+
"""

OUTPUT_FIELDS = [
    "sentence_id", "company", "year", "report_type", "sentence_text",
    "llm_label", "llm_confidence", "llm_top_features", "llm_gw_risk",
    "kee_label", "kee_confidence", "kee_notes", "agreement_yn",
    "annotated_at",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_sentences():
    with open(INPUT_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing(path):
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[row["sentence_id"]] = row
    return done


def save(path, annotations):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(annotations)


def append_log(msg: str):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("[{}] {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M"), msg))


def print_progress(done: int, total: int, agreed: int):
    bar_len = 30
    filled = int(bar_len * done / max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = done / max(total, 1) * 100
    agree_pct = agreed / max(done, 1) * 100
    print("\n  Progress: [{}] {}/{} ({:.0f}%)  |  Agreement: {}/{} ({:.0f}%)".format(
        bar, done, total, pct, agreed, done, agree_pct))


# ---------------------------------------------------------------------------
# MAIN ANNOTATION LOOP
# ---------------------------------------------------------------------------

def annotate(session: int, start: int, end: int,
             resume: bool, express: bool, rater: str):

    out_path = os.path.join(
        OUTPUT_DIR, "{}_session{}.csv".format(rater, session))

    all_sents = load_sentences()
    done      = load_existing(out_path)
    annotations = list(done.values())

    # Select window (1-indexed)
    batch = all_sents[start - 1:end]
    total = len(batch)

    if not batch:
        print("No sentences in range {}-{} (total: {})".format(
            start, end, len(all_sents)))
        return

    print("\n" + "═" * 72)
    print("  HYBRID ANNOTATION TOOL  (xNLP-Assisted)")
    print("  Rater: {}  |  Session: {}  |  Range: S{:04d}–S{:04d}".format(
        rater, session, start, end))
    print("  Mode: {}  |  Previously done: {}".format(
        "EXPRESS" if express else "INTERACTIVE", len(done)))
    print("═" * 72)
    if not express:
        print(SCHEMA_QUICK)

    history  = []
    agreed_n = sum(1 for a in annotations if a.get("agreement_yn") == "Y")
    i        = 0

    while i < len(batch):
        sent   = batch[i]
        sid    = sent["sentence_id"]
        qual   = sent.get("quality", "ok")
        text   = sent["sentence_text"]
        co     = sent["company"]
        yr     = sent["year"]
        rtype  = sent["report_type"]

        # Resume: skip already annotated
        if resume and sid in done:
            i += 1
            continue

        # ── Run LLM feature analysis ──────────────────────────────────────
        llm_label, llm_conf, top3, gw_risk, gw_label = suggest_label(
            text, qual)
        feat_csv = format_features_for_csv(top3)

        # ── Display analysis panel ────────────────────────────────────────
        display_analysis(sid, co, yr, rtype, qual, text,
                         llm_label, llm_conf, top3, gw_risk, gw_label)

        # ── Express mode: auto-accept ─────────────────────────────────────
        if express:
            ann = {
                "sentence_id":    sid,
                "company":        co,
                "year":           yr,
                "report_type":    rtype,
                "sentence_text":  text,
                "llm_label":      llm_label,
                "llm_confidence": str(llm_conf),
                "llm_top_features": feat_csv,
                "llm_gw_risk":    str(gw_risk),
                "kee_label":      llm_label,
                "kee_confidence": str(llm_conf),
                "kee_notes":      "[express-accepted]",
                "agreement_yn":   "Y",
                "annotated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            annotations.append(ann)
            done[sid] = ann
            agreed_n += 1
            i += 1
            print_progress(len([a for a in annotations
                                 if a["sentence_id"].startswith("S")]),
                           total, agreed_n)
            continue

        # ── Interactive mode ──────────────────────────────────────────────
        print("\n  Speed tip: Y=accept | N=override | ?=unsure | B=undo | Q=quit | H=help")
        while True:
            raw = input("\n  → Accept LLM label [{}]? [Y/N/?/B/Q/H]: ".format(
                llm_label)).strip().upper()

            if raw == "H":
                print(SCHEMA_QUICK)
                continue

            elif raw == "Q":
                save(out_path, annotations)
                append_log("Session {} quit at {}. Saved {} annotations.".format(
                    session, sid, len(annotations)))
                print("\n  Saved {} annotations → {}".format(
                    len(annotations), out_path))
                _print_session_summary(annotations, out_path)
                return

            elif raw == "B":
                if history:
                    psid, _ = history.pop()
                    old_ann = done.pop(psid, None)
                    annotations = [a for a in annotations
                                   if a["sentence_id"] != psid]
                    if old_ann and old_ann.get("agreement_yn") == "Y":
                        agreed_n = max(0, agreed_n - 1)
                    i -= 1
                    print("  [Undid annotation for {}]".format(psid))
                    break
                else:
                    print("  [Nothing to undo]")

            elif raw == "Y":
                # Accept LLM label
                c = input("  Your confidence [1=sure/2=moderate/3=unsure]: ").strip()
                kee_conf = c if c in ("1", "2", "3") else str(llm_conf)
                ann = _build_ann(
                    sid, co, yr, rtype, text, feat_csv,
                    llm_label, llm_conf, gw_risk,
                    kee_label=llm_label,
                    kee_conf=kee_conf,
                    kee_notes="",
                    agreement="Y",
                )
                _commit(annotations, done, history, ann, sid)
                agreed_n += 1
                i += 1; break

            elif raw == "N":
                # Override
                while True:
                    kl = input("  Your label [V/S/N/?]: ").strip().upper()
                    if kl in ("V", "S", "N", "?"):
                        break
                    print("  [Use V S N ?]")
                c = input("  Your confidence [1/2/3]: ").strip()
                kee_conf = c if c in ("1", "2", "3") else "2"
                note = input("  Notes (Enter to skip): ").strip()
                agree = "Y" if kl == llm_label else "N"
                ann = _build_ann(
                    sid, co, yr, rtype, text, feat_csv,
                    llm_label, llm_conf, gw_risk,
                    kee_label=kl, kee_conf=kee_conf,
                    kee_notes=note, agreement=agree,
                )
                _commit(annotations, done, history, ann, sid)
                if agree == "Y":
                    agreed_n += 1
                print("  [Override: {} → {}  Agreement={}]".format(
                    llm_label, kl, agree))
                i += 1; break

            elif raw == "?":
                c = input("  Your confidence [1/2/3]: ").strip()
                kee_conf = c if c in ("1", "2", "3") else "3"
                note = input("  Notes: ").strip()
                agree = "Y" if llm_label == "?" else "N"
                ann = _build_ann(
                    sid, co, yr, rtype, text, feat_csv,
                    llm_label, llm_conf, gw_risk,
                    kee_label="?", kee_conf=kee_conf,
                    kee_notes=note, agreement=agree,
                )
                _commit(annotations, done, history, ann, sid)
                i += 1; break

            else:
                print("  [Invalid. Use Y N ? B Q H]")

        # Progress update
        done_count = i
        print_progress(done_count, total, agreed_n)

        # Auto-save every 10 annotations
        if done_count % 10 == 0 and done_count > 0:
            save(out_path, annotations)
            print("  [Auto-saved at sentence {}/{}]".format(done_count, total))

    # Final save
    save(out_path, annotations)
    append_log("Session {} complete. {} annotations saved.".format(
        session, len(annotations)))
    _print_session_summary(annotations, out_path)


def _build_ann(sid, co, yr, rtype, text, feat_csv,
               llm_lbl, llm_conf, gw_risk,
               kee_label, kee_conf, kee_notes, agreement):
    return {
        "sentence_id":      sid,
        "company":          co,
        "year":             yr,
        "report_type":      rtype,
        "sentence_text":    text,
        "llm_label":        llm_lbl,
        "llm_confidence":   str(llm_conf),
        "llm_top_features": feat_csv,
        "llm_gw_risk":      str(gw_risk),
        "kee_label":        kee_label,
        "kee_confidence":   kee_conf,
        "kee_notes":        kee_notes,
        "agreement_yn":     agreement,
        "annotated_at":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _commit(annotations, done, history, ann, sid):
    annotations.append(ann)
    done[sid] = ann
    history.append((sid, ann))


def _print_session_summary(annotations, out_path):
    from collections import Counter
    kee_dist = Counter(a["kee_label"] for a in annotations)
    llm_dist = Counter(a["llm_label"] for a in annotations)
    agreed   = sum(1 for a in annotations if a["agreement_yn"] == "Y")
    total    = len(annotations)
    agree_pct = agreed / max(total, 1) * 100

    gw_high = sum(1 for a in annotations
                  if float(a.get("llm_gw_risk", 0)) >= 0.65)

    print("\n" + "═" * 72)
    print("  SESSION SUMMARY")
    print("  Total annotated : {}".format(total))
    print("  Kee labels      : {}".format(dict(kee_dist)))
    print("  LLM labels      : {}".format(dict(llm_dist)))
    print("  Agreement       : {}/{} ({:.1f}%)".format(agreed, total, agree_pct))
    print("  GW HIGH risk    : {} sentences".format(gw_high))
    print("  Saved to        : {}".format(out_path))
    print("═" * 72)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="xNLP Hybrid Annotation Tool (LLM-Assisted Solo)")
    parser.add_argument("--rater",   default="kee",
                        help="Rater ID (default: kee)")
    parser.add_argument("--session", type=int, default=1,
                        help="Session number (default: 1)")
    parser.add_argument("--start",   type=int, default=1,
                        help="Start sentence number 1-indexed (default: 1)")
    parser.add_argument("--end",     type=int, default=25,
                        help="End sentence number 1-indexed (default: 25)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip already-annotated sentences")
    parser.add_argument("--express", action="store_true",
                        help="Auto-accept all LLM labels (no interactive prompts)")
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        print("ERROR: {} not found. Run sentence_segmenter.py first.".format(INPUT_CSV))
        sys.exit(1)

    annotate(args.session, args.start, args.end,
             args.resume, args.express, args.rater)
