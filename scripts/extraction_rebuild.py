"""
Extraction Rebuild — Phase 2
==============================
Replaces the old PDF→text pipeline with a column-aware extraction.

OLD PIPELINE (broken):
    unknown extract_text.py  →  data/extracted/<company>/<year>.txt  (garbled, multicolumn bleed)
    section_extractor.py     →  data/brsr_sections/<company>/<year>.txt  (operating on garbled text)

NEW PIPELINE (this script):
    raw PDF → pdfplumber page scan → section pages found
            → Method D split-column extraction on section pages
            → data/brsr_sections/<company>/<year>.txt  (clean, column-aware)

Strategy
--------
1. Parse raw PDF filename:  raw_<company>_<year>.pdf  →  (company, year)
2. Fast-scan pages with basic extract_text() to locate section start/end
   (same anchor patterns as section_extractor.py, but applied per-page)
3. Extract section pages using Method D (split-column via column_extractor.py)
4. Write clean section text to data/brsr_sections/<company>/<year>.txt
5. Log stats: pages found, extraction method, ESG density

Outputs
-------
    data/brsr_sections/<company>/<year>.txt   — clean section text (overwrites old)
    logs/extraction_rebuild_log.txt           — per-file status log
    logs/extraction_rebuild_stats.txt         — summary stats table
"""

import os
import re
import sys
from datetime import datetime

import pdfplumber

# ---------------------------------------------------------------------------
# PATH SETUP (run from project root)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from column_extractor import extract_page, detect_column_split, is_rotated_text

RAW_DIR    = "data/raw"
OUT_DIR    = "data/brsr_sections"
LOG_DIR    = "logs"
LOG_PATH   = os.path.join(LOG_DIR, "extraction_rebuild_log.txt")
STATS_PATH = os.path.join(LOG_DIR, "extraction_rebuild_stats.txt")

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# COMPANY NAME ALIASES  (raw folder names → canonical)
# ---------------------------------------------------------------------------
COMPANY_ALIAS = {
    "ultratect": "ultratech",
}

# ---------------------------------------------------------------------------
# SECTION ANCHOR PATTERNS (ported from section_extractor.py, page-level)
# Each strategy is a list of regex patterns matched against full-page text.
# Priority: S1 > S2 > S3
# ---------------------------------------------------------------------------

# S1: BRSR mandatory / transition (2022+)
BRSR_ANCHORS = [
    re.compile(r"section\s+a\s*:\s*general\s+disclos",           re.IGNORECASE),
    re.compile(r"business\s+responsibility\s+and\s+sustainability\s+report\s+202", re.IGNORECASE),
    re.compile(r"statutory\s+reports.*business\s+responsibility", re.IGNORECASE),
    re.compile(r"brsr\s+202[2-9]",                               re.IGNORECASE),
]

# S2: BRR standalone (pre-2022)
BRR_ANCHORS = [
    re.compile(r"^\s*business\s+responsibility\s+report\s*$",           re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*business\s+responsibility\s+report\s*\(brr\)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*section\s+a\s*:\s*general\s+information\s+about",  re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*section\s+e\s*:\s*principle\s+wise",               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*principle.wise\s+\(as\s+per\s+nvgs\)",             re.IGNORECASE | re.MULTILINE),
]

# S3: Integrated Report / Natural Capital (Tata pre-2022)
INTEGRATED_ANCHORS = [
    re.compile(r"^\s*natural\s+capital\s*$",                         re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*environment\s+and\s+decarbonisation\s*$",       re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*environmental\s+performance\s*$",               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*environmental\s+initiatives\s*$",               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*reduction\s+of\s+emissions\s+and\s+discharges", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*environment\s+[&and]+\s+water\s*$",             re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*esg\s+factsheet",                               re.IGNORECASE | re.MULTILINE),
]

# End-of-section markers
END_ANCHORS = [
    re.compile(r"independent\s+auditor",                                           re.IGNORECASE),
    re.compile(r"standalone\s+financial\s+statements",                             re.IGNORECASE),
    re.compile(r"standalone\s+balance\s+sheet",                                    re.IGNORECASE),
    re.compile(r"notes\s+to\s+the\s+(standalone|consolidated|financial)\s+.{0,20}statements", re.IGNORECASE),
    re.compile(r"consolidated\s+balance\s+sheet",                                  re.IGNORECASE),
    re.compile(r"report\s+on\s+corporate\s+governance",                            re.IGNORECASE),
    re.compile(r"^\s*directors['\s]+report\s*$",                                   re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*board.s\s+report\s*$",                                        re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*annexure\s*-\s*[a-z]\s+to\s+directors",                      re.IGNORECASE | re.MULTILINE),
]

# ESG keywords for density scoring (used to validate anchor hits vs TOC references)
ESG_KEYWORDS = [
    "emission", "carbon", "ghg", "scope 1", "scope 2", "scope 3",
    "greenhouse", "renewable", "energy consumption", "water withdrawal",
    "waste", "effluent", "biodiversity", "co2", "net zero", "net-zero",
    "decarbonisation", "sustainability", "environment", "climate",
    "principle 6", "essential indicator", "brsr", "brr", "ngrbc",
    "social", "governance", "disclosure",
]


# ---------------------------------------------------------------------------
# PDF FILENAME PARSER
# ---------------------------------------------------------------------------

def parse_pdf_name(filename):
    """
    Parse  raw_<company>_<year>.pdf  →  (company: str, year: int)
    Handles the ultratect → ultratech alias.
    Returns None if filename doesn't match the expected pattern.
    """
    m = re.match(r"raw_([a-z]+)_(\d{4})\.pdf$", filename, re.IGNORECASE)
    if not m:
        return None
    raw_company = m.group(1).lower()
    year        = int(m.group(2))
    company     = COMPANY_ALIAS.get(raw_company, raw_company)
    return company, year


# ---------------------------------------------------------------------------
# SECTION PAGE LOCATOR (fast scan, basic extract_text per page)
# ---------------------------------------------------------------------------

def esg_density(text):
    """Count ESG keyword hits in text (used to rank candidate anchor pages)."""
    tl = text.lower()
    return sum(1 for kw in ESG_KEYWORDS if kw in tl)


def is_end_page(text):
    """Return True if this page contains an end-of-section marker."""
    return any(p.search(text) for p in END_ANCHORS)


def scan_for_section(pdf, year):
    """
    Fast-scan PDF pages to find the section start/end page indices.

    Strategy:
    1. Start scanning from page 15% into document (skip cover, ToC)
    2. Try BRSR anchors first (S1), then BRR (S2), then Integrated (S3)
    3. For each candidate anchor hit, score ESG density to reject ToC references
    4. End page = first page after start with an end-of-section marker

    Returns:
        (start_page: int, end_page: int, strategy: str)
        or None if no section found
    """
    total      = len(pdf.pages)
    scan_from  = max(0, int(total * 0.15))

    # Determine which strategy set to try based on year
    if year >= 2022:
        strategy_order = [
            ("S1-BRSR",       BRSR_ANCHORS),
            ("S2-BRR",        BRR_ANCHORS),
            ("S3-INTEGRATED", INTEGRATED_ANCHORS),
        ]
    else:
        strategy_order = [
            ("S2-BRR",        BRR_ANCHORS),
            ("S3-INTEGRATED", INTEGRATED_ANCHORS),
            ("S1-BRSR",       BRSR_ANCHORS),   # fallback
        ]

    # Cache page texts to avoid re-extraction
    page_texts = {}

    def get_text(idx):
        if idx not in page_texts:
            try:
                page_texts[idx] = pdf.pages[idx].extract_text() or ""
            except Exception:
                page_texts[idx] = ""
        return page_texts[idx]

    # Collect all anchor hits across all strategies
    # Format: (strategy_name, page_idx, esg_score)
    candidates = []

    for strat_name, anchors in strategy_order:
        for i in range(scan_from, total):
            text = get_text(i)
            for pat in anchors:
                if pat.search(text):
                    score = esg_density(text)
                    candidates.append((strat_name, i, score))
                    break  # one anchor match per page per strategy is enough

        # If this strategy found a strong hit (ESG score >= 3), stop
        strong = [c for c in candidates if c[0] == strat_name and c[2] >= 3]
        if strong:
            break

    # Also scan early pages if nothing found
    if not candidates:
        for i in range(0, scan_from):
            text = get_text(i)
            for strat_name, anchors in strategy_order:
                for pat in anchors:
                    if pat.search(text):
                        score = esg_density(text)
                        candidates.append((strat_name, i, score))
                        break

    if not candidates:
        return None

    # Pick best: highest ESG density, then lowest page index
    candidates.sort(key=lambda c: (-c[2], c[1]))
    best_strat, start_page, start_score = candidates[0]

    # Find end page (scan up to 60 pages ahead)
    end_page = None
    limit    = min(start_page + 60, total)
    for i in range(start_page + 1, limit):
        if is_end_page(get_text(i)):
            end_page = i
            break

    if end_page is None:
        end_page = min(start_page + 40, total - 1)

    return start_page, end_page, best_strat, start_score


# ---------------------------------------------------------------------------
# METHOD D EXTRACTION (column-aware, page by page)
# ---------------------------------------------------------------------------

def extract_section_column_aware(pdf, start_page, end_page):
    """
    Extract text from pages [start_page, end_page) using Method D:
    split-column extraction with rotated-sidebar filtering.

    Returns:
        (section_text: str, page_meta: list[dict])
    """
    parts      = []
    page_metas = []

    for idx in range(start_page, end_page + 1):
        page = pdf.pages[idx]
        try:
            text, meta = extract_page(page, include_right=True)
            meta["page_index"] = idx
            parts.append(text)
            page_metas.append(meta)
        except Exception as e:
            # Fallback to basic extract_text on page error
            try:
                fallback = page.extract_text() or ""
                parts.append(fallback)
                page_metas.append({
                    "page_index":      idx,
                    "split_x":         page.width / 2,
                    "right_discarded": False,
                    "error":           str(e),
                    "fallback":        True,
                })
            except Exception:
                pass

    return "\n\n".join(parts).strip(), page_metas


# ---------------------------------------------------------------------------
# MAIN REBUILD LOOP
# ---------------------------------------------------------------------------

def run_rebuild():
    os.makedirs(LOG_DIR, exist_ok=True)

    log_entries  = [f"=== Extraction Rebuild Run: {datetime.now().isoformat()} ===\n"]
    stats_rows   = []
    stats        = {"ok": 0, "failed": 0, "total": 0}

    # Collect all PDFs
    pdf_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".pdf"))

    print("=" * 70)
    print(f"EXTRACTION REBUILD — {len(pdf_files)} PDFs")
    print("=" * 70)

    for filename in pdf_files:
        parsed = parse_pdf_name(filename)
        if parsed is None:
            print(f"[SKIP] {filename}: does not match raw_<company>_<year>.pdf")
            continue

        company, year = parsed
        stats["total"] += 1
        pdf_path = os.path.join(RAW_DIR, filename)
        regime   = "BRSR" if year >= 2022 else "BRR"

        print(f"\n[{stats['total']:02d}] {company}/{year} ({regime}) — {filename}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Step 1: Locate section
                result = scan_for_section(pdf, year)

                if result is None:
                    raise ValueError("Section not found — no anchor matched")

                start_page, end_page, strategy, esg_score = result
                section_pages = end_page - start_page + 1

                print(f"     Section: pages {start_page+1}–{end_page+1} "
                      f"({section_pages} pages) | {strategy} | ESG score={esg_score}")

                # Step 2: Extract section using Method D
                section_text, page_metas = extract_section_column_aware(
                    pdf, start_page, end_page)

                # Check for rotated-sidebar pages
                rotated_count = sum(1 for m in page_metas if m.get("right_discarded"))
                fallback_count = sum(1 for m in page_metas if m.get("fallback"))

        except Exception as e:
            print(f"     [FAIL] {e}")
            log_entries.append(
                f"[FAIL] company={company} year={year} regime={regime} "
                f"error={e}")
            stats["failed"] += 1

            # Write failure placeholder
            out_dir  = os.path.join(OUT_DIR, company)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{year}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"[EXTRACTION FAILED — requires manual review]\n"
                        f"company={company}, year={year}, regime={regime}\n"
                        f"error={e}\n")

            stats_rows.append({
                "company": company, "year": year, "regime": regime,
                "status": "FAIL", "pages": 0, "strategy": "NONE",
                "size_kb": 0, "esg_score": 0, "rotated": 0,
            })
            continue

        # Step 3: Validate minimum length
        if len(section_text.strip()) < 200:
            print(f"     [WARN] Section too short ({len(section_text)} chars) — writing placeholder")
            section_text = (
                f"[EXTRACTION WARNING — section too short]\n"
                f"company={company}, year={year}, regime={regime}\n"
                f"strategy={strategy}, esg_score={esg_score}\n"
            )

        # Step 4: Write output
        out_dir  = os.path.join(OUT_DIR, company)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{year}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(section_text)

        # Stats
        line_count = len(section_text.splitlines())
        byte_count = len(section_text.encode("utf-8"))
        size_kb    = byte_count // 1024

        print(f"     [OK]  {line_count} lines / {size_kb} KB | "
              f"rotated_discarded={rotated_count} | fallbacks={fallback_count}")

        log_entries.append(
            f"[OK]  company={company} year={year} regime={regime} "
            f"strategy={strategy} esg={esg_score} pages={section_pages} "
            f"lines={line_count} size_kb={size_kb} "
            f"rotated={rotated_count} fallbacks={fallback_count}")

        stats_rows.append({
            "company":  company,
            "year":     year,
            "regime":   regime,
            "status":   "OK",
            "pages":    section_pages,
            "strategy": strategy,
            "size_kb":  size_kb,
            "esg_score": esg_score,
            "rotated":  rotated_count,
        })
        stats["ok"] += 1

    # ---------------------------------------------------------------------------
    # WRITE LOG
    # ---------------------------------------------------------------------------
    summary = (f"\nSUMMARY: total={stats['total']} "
               f"ok={stats['ok']} failed={stats['failed']}\n")
    log_entries.append(summary)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries))

    # ---------------------------------------------------------------------------
    # WRITE STATS TABLE
    # ---------------------------------------------------------------------------
    col = "{:<12} {:>5} {:>8} {:>18} {:>8} {:>8} {:>8} {:>8}"
    header = col.format("company", "year", "regime", "strategy",
                        "pages", "size_kb", "esg", "rotated")
    divider = "-" * len(header)

    stats_lines = [
        f"=== Extraction Rebuild Stats: {datetime.now().isoformat()} ===\n",
        header,
        divider,
    ]
    for r in sorted(stats_rows, key=lambda x: (x["company"], x["year"])):
        stats_lines.append(col.format(
            r["company"], r["year"], r["regime"], r["strategy"],
            r["pages"], r["size_kb"], r["esg_score"], r["rotated"],
        ))
    stats_lines.append(divider)
    stats_lines.append(f"TOTAL: {stats['ok']}/{stats['total']} OK, {stats['failed']} FAILED\n")

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))

    # ---------------------------------------------------------------------------
    # CONSOLE SUMMARY
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"REBUILD COMPLETE: {stats['ok']}/{stats['total']} extracted successfully")
    print(f"{'='*70}")
    print(header)
    print(divider)
    for r in sorted(stats_rows, key=lambda x: (x["company"], x["year"])):
        status_tag = "[OK]  " if r["status"] == "OK" else "[FAIL]"
        print(f"{status_tag} " + col.format(
            r["company"], r["year"], r["regime"], r["strategy"],
            r["pages"], r["size_kb"], r["esg_score"], r["rotated"],
        ))
    print(divider)
    print(f"Log   → {LOG_PATH}")
    print(f"Stats → {STATS_PATH}")


if __name__ == "__main__":
    run_rebuild()
