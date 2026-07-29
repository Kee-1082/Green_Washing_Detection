"""
Feature Extractor for Explainable NLP Annotation
==================================================
Computes 5 weighted linguistic features per sentence and produces
a rule-based label suggestion (V/S/N/?) + confidence score.

Features (aligned with the ArXiv 2024 explainable NLP paper):
  1. vague_adj_ratio   - soft commitment verbs & hedging adjectives
  2. quantifier_count  - numbers, %, years, physical units detected
  3. verb_strength     - past-tense specific action vs modal/hedging
  4. target_year       - presence of deadline years (FY2030, by 2050...)
  5. specific_tech     - named technologies, locations, certifications

Each feature returns a score in [0.0, 1.0].
The label is determined by a weighted decision rule — fully transparent,
no black-box ML.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# FEATURE WORD LISTS
# ---------------------------------------------------------------------------

HEDGE_VERBS = [
    "aim", "aims", "aimed",
    "strive", "strives", "strived",
    "seek", "seeks", "sought",
    "endeavour", "endeavours", "endeavoured",
    "intend", "intends", "intended",
    "aspire", "aspires", "aspired",
    "promote", "promotes", "promoted",
    "focus on", "focused on",
    "work towards", "working towards",
    "committed to", "commit to",
    "dedicated to",
    "targeting",   # bare 'plan/plans' excluded (false-positives on 'plant')
    "hope", "hopes",
    "ensure", "ensures",
    "planning to", "plans to",  # only match 'plans to' not bare 'plans'
]

HEDGE_ADJECTIVES = [
    "significant", "significant", "substantially",
    "meaningful", "meaningfully",
    "considerable", "considerably",
    "better", "improved", "enhanced", "greater",
    "responsible", "sustainable", "green",
    "low-carbon", "environmentally friendly",
    "comprehensive", "holistic",
    "long-term", "long term",
]

STRONG_VERBS = [
    "reduced", "reduce by", "cut", "decreased",
    "achieved", "commissioned", "installed",
    "deployed", "launched", "completed",
    "recycled", "recovered", "diverted",
    "saved", "avoided", "offset",
    "certified", "audited", "measured",
    "invested", "allocated",
    "disclosed", "reported",
    "replaced", "retrofitted", "upgraded",
]

# Named technologies relevant to cement/steel sector
SPECIFIC_TECH = [
    # Technologies
    "ccus", "carbon capture", "waste heat recovery", "whr",
    "solar pv", "solar plant", "wind energy", "wind farm",
    "blast furnace", "electric arc furnace", "eaf",
    "coke dry quenching", "cdq",
    "flue gas", "scrubber", "electrostatic precipitator", "esp",
    "reverse osmosis", "membrane bioreactor",
    "biogas plant", "landfill gas",
    "oxy-fuel combustion", "hydrogen blending",
    # Named plants / locations
    "vijayanagar", "dolvi", "jamshedpur", "kalinganagar",
    "port talbot", "ijmuiden", "tata steel europe",
    "ultratech cement", "birla white",
    "jsw steel", "jsw energy",
    # Standards / certifications
    "iso 14001", "iso 50001", "gri", "tcfd", "sasb",
    "science based targets", "sbti",
    "carbon disclosure project", "cdp",
    # Units that signal specific measurement (in prose)
    "mw solar", "mw wind", "mw renewable",
    "tco2", "tco2e", "scope 1", "scope 2", "scope 3",
]

# Physical / financial unit patterns
UNIT_PATTERNS = [
    r"\d+[\.,]?\d*\s*%",                      # percentages
    r"\d+[\.,]?\d*\s*(mw|gw|kwh|gwh|gj|tj)", # energy units
    r"\d+[\.,]?\d*\s*(m3|kl|ml|litre)",       # water units
    r"\d+[\.,]?\d*\s*(tonne|ton|mt|mnt|kt)",  # mass units
    r"\d+[\.,]?\d*\s*(tco2|tco2e|co2e)",      # carbon units
    r"\d+[\.,]?\d*\s*(crore|lakh|million|billion)",  # financial
    r"\brs\.?\s*\d+",                         # rupee amounts
    r"fy\s*20\d\d",                           # financial years
    r"\bfy\d{2}[-–]\d{2}\b",                  # FY22-23 style
    r"\bby\s+20[2-5]\d\b",                    # "by 2030" deadlines
    r"\b20[2-5]\d\s+baseline\b",              # "2019 baseline"
    r"\b\d{1,3}(,\d{3})+\b",                  # large numbers with commas
]

# Target year patterns (deadlines / baselines)
TARGET_YEAR_PATTERNS = [
    r"\bby\s+(fy)?\s*20[2-5]\d\b",            # "by 2030", "by FY2030"
    r"\bby\s+fy\d{2}[-–]\d{2}\b",             # "by FY30-31"
    r"\b20[2-5]\d\s+(target|goal|baseline)\b", # "2030 target"
    r"\bnet[-\s]?zero\s+by\s+20\d{2}\b",       # "net-zero by 2050"
    r"\bcarbon\s+neutral\s+by\s+20\d{2}\b",    # "carbon neutral by 2070"
    r"\bby\s+the\s+end\s+of\s+fy\s*20\d{2}",  # "by the end of FY2025"
]


# ---------------------------------------------------------------------------
# FEATURE COMPUTATION
# ---------------------------------------------------------------------------

@dataclass
class FeatureResult:
    name: str
    score: float        # 0.0 to 1.0
    signals: List[str]  # matched tokens/patterns
    interpretation: str


def feat_vague_adj_ratio(text_lower: str, tokens: List[str]) -> FeatureResult:
    """Ratio of hedge words to total content words. Uses word boundaries to avoid false positives."""
    hits = []
    for hw in HEDGE_VERBS + HEDGE_ADJECTIVES:
        # Use word-boundary aware matching for single words
        if " " in hw:
            if hw in text_lower:
                hits.append(hw)
        else:
            # Word boundary check: the term must not be part of a longer word
            if re.search(r'\b' + re.escape(hw) + r'\b', text_lower):
                hits.append(hw)
    content_words = [t for t in tokens if len(t) > 3]
    ratio = min(len(hits) / max(len(content_words), 1), 1.0)
    score = min(ratio * 4, 1.0)  # normalise: >=4 hedge words → score=1.0
    interp = (
        "High hedge density → strong vague signal" if score > 0.5 else
        "Moderate hedging → lean vague" if score > 0.2 else
        "Low hedging → not a strong vague signal"
    )
    return FeatureResult("vague_adj_ratio", round(score, 3),
                         hits[:3], interp)


def feat_quantifier_count(text_lower: str) -> FeatureResult:
    """Count of numeric / unit patterns — substantive signal."""
    hits = []
    for pat in UNIT_PATTERNS:
        matches = re.findall(pat, text_lower, re.IGNORECASE)
        hits.extend(matches)
    # Deduplicate
    hits = list(dict.fromkeys(str(h) for h in hits))
    score = min(len(hits) / 3.0, 1.0)   # >=3 unique patterns → score=1.0
    interp = (
        "Multiple quantifiers → strong substantive signal" if score > 0.6 else
        "Some quantifiers → lean substantive" if score > 0.2 else
        "No quantifiers detected → vague/numeric flag"
    )
    return FeatureResult("quantifier_count", round(score, 3),
                         hits[:4], interp)


def feat_verb_strength(text_lower: str) -> FeatureResult:
    """Ratio of strong/specific verbs to total verb-like tokens."""
    strong_hits = [v for v in STRONG_VERBS if v in text_lower]
    hedge_hits  = [v for v in HEDGE_VERBS   if v in text_lower]
    total = len(strong_hits) + len(hedge_hits)
    if total == 0:
        score = 0.1  # no verbs → likely numeric
        interp = "No verifiable action verb → likely N or incomplete sentence"
    elif strong_hits and not hedge_hits:
        score = 0.85
        interp = "Strong past-tense action verb → substantive signal"
    elif strong_hits and hedge_hits:
        score = 0.45
        interp = "Mix of strong + hedge verbs → could be V or S"
    else:
        score = 0.15
        interp = "Only hedging verbs → vague signal"
    signals = (strong_hits + hedge_hits)[:3]
    return FeatureResult("verb_strength", round(score, 3), signals, interp)


def feat_target_year(text_lower: str) -> FeatureResult:
    """Presence of deadline years or baseline references."""
    hits = []
    for pat in TARGET_YEAR_PATTERNS:
        matches = re.findall(pat, text_lower, re.IGNORECASE)
        hits.extend(matches)
    hits = list(dict.fromkeys(str(h) for h in hits))
    score = min(len(hits) / 2.0, 1.0)
    interp = (
        "Clear deadline/target year → substantive even with hedge verb" if score > 0.4 else
        "Vague time reference or none → no deadline signal"
    )
    return FeatureResult("target_year", round(score, 3), hits[:3], interp)


def feat_specific_tech(text_lower: str) -> FeatureResult:
    """Named technology, location, or certification standard."""
    hits = [t for t in SPECIFIC_TECH if t in text_lower]
    score = min(len(hits) / 2.0, 1.0)
    interp = (
        "Named tech/location/standard → substantive signal" if hits else
        "No named technology or standard → generic claim"
    )
    return FeatureResult("specific_tech", round(score, 3), hits[:3], interp)


# ---------------------------------------------------------------------------
# COMPOSITE LABEL DECISION
# ---------------------------------------------------------------------------

# Feature weights for label decision (sum to 1.0)
WEIGHTS = {
    # Decision weights — ordered by analyst-validated discriminative power:
    #   1. quantifier_count  HIGHEST  number+unit+% almost always → S
    #   2. specific_tech      HIGH    named tech → S; generic terms → V
    #   3. vague_adj_ratio   MEDIUM   supporting signal only
    #   4. target_year        LOW     only matters combined with quant/tech
    #   5. verb_strength     LOWEST   least discriminative alone
    "vague_adj_ratio":  0.15,
    "quantifier_count": 0.40,
    "verb_strength":    0.08,
    "target_year":      0.12,
    "specific_tech":    0.25,
}


def is_table_row(text: str) -> bool:
    """Detect data table row: alpha_ratio < 0.45 and no full verb phrase."""
    alpha = sum(1 for c in text if c.isalpha())
    ratio = alpha / max(len(text), 1)
    has_verb_phrase = bool(re.search(
        r'\b(is|are|was|were|has|have|had|will|would|should|reduced|achieved|'
        r'installed|committed|aims|strives)\b', text, re.IGNORECASE))
    return ratio < 0.45 and not has_verb_phrase


def suggest_label(text: str, quality_flag: str = "ok") -> Tuple[
        str, int, List[FeatureResult], float, str]:
    """
    Returns: (label, confidence, features, gwash_risk_score, gwash_risk_label)

    Decision cascade (analyst-validated priority order):
      1. quantifier_count > 0  → S  (number + unit + % is primary signal)
      2. specific_tech > 0.4   → S  (named tech = substantive, no quant needed)
      3. vague_adj_ratio > 0.5 → V  (high hedge ratio = vague)
      4. target_year + verb as tiebreaker
    """
    text_lower = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text_lower)

    # --- Shortcut: table row ---
    if quality_flag == "table_row" or is_table_row(text):
        qf = feat_quantifier_count(text_lower)
        return ("N", 1,
                [qf,
                 FeatureResult("verb_strength", 0.05, [], "No verb → data row"),
                 FeatureResult("vague_adj_ratio", 0.0, [], "N/A for data rows")],
                0.0, "LOW")

    # --- Compute all features ---
    f_vague = feat_vague_adj_ratio(text_lower, tokens)
    f_quant = feat_quantifier_count(text_lower)
    f_verb  = feat_verb_strength(text_lower)
    f_year  = feat_target_year(text_lower)
    f_tech  = feat_specific_tech(text_lower)

    features = [f_vague, f_quant, f_verb, f_year, f_tech]

    # --- PRIORITY CASCADE (analyst decision rule) ---
    #
    # Step 1: quantifier_count (HIGHEST)
    # Any number + unit + % pattern → almost always S
    if f_quant.score > 0.0:
        label = "S"
        # High confidence if strong quant, medium if borderline
        conf  = 1 if f_quant.score >= 0.5 else 2

    # Step 2: specific_tech (HIGH)
    # Named technology even without a number → S
    elif f_tech.score > 0.4:
        label = "S"
        conf  = 1 if f_tech.score >= 0.7 else 2

    # Step 3: vague_adj_ratio (MEDIUM)
    # High hedge language → V, but only as supporting signal
    elif f_vague.score > 0.5:
        label = "V"
        conf  = 1 if f_vague.score >= 0.7 else 2

    # Step 4: tiebreaker — target_year + verb_strength
    else:
        combined = (
            f_year.score  * WEIGHTS["target_year"] +
            f_verb.score  * WEIGHTS["verb_strength"] +
            (1 - f_vague.score) * WEIGHTS["vague_adj_ratio"]
        )
        if combined >= 0.12:          # both year + strong verb present
            label, conf = "S", 2
        elif f_vague.score > 0.25:    # some hedging → lean V
            label, conf = "V", 2
        else:
            label, conf = "?", 3      # genuinely ambiguous

    # --- Greenwashing risk (vague-weighted, quantifier-penalised) ---
    gw_risk = f_vague.score * 0.6 + (1 - f_quant.score) * 0.4
    if gw_risk >= 0.65:
        gw_label = "HIGH"
    elif gw_risk >= 0.40:
        gw_label = "MODERATE"
    else:
        gw_label = "LOW"

    # Sort features by relevance for display (top 3 most informative)
    top3 = sorted(features, key=lambda f: abs(f.score - 0.5), reverse=True)[:3]

    return label, conf, top3, round(gw_risk, 3), gw_label


def format_features_for_csv(features: List[FeatureResult]) -> str:
    """Compact CSV-safe feature string."""
    parts = []
    for f in features:
        sig = "|".join(f.signals) if f.signals else "NONE"
        parts.append("{}={:.2f}[{}]".format(f.name, f.score, sig))
    return "; ".join(parts)


def display_analysis(sent_id: str, company: str, year: str,
                     report_type: str, quality: str,
                     text: str, label: str, conf: int,
                     features: List[FeatureResult],
                     gw_risk: float, gw_label: str) -> None:
    """Print the annotator-facing analysis panel (ASCII-safe for Windows)."""
    LABEL_NAMES = {"V": "VAGUE", "S": "SUBSTANTIVE", "N": "NUMERIC-FIN", "?": "UNSURE"}
    CONF_NAMES  = {1: "VERY SURE", 2: "MODERATE", 3: "UNSURE"}
    GW_FLAGS    = {"HIGH": "[!!!]", "MODERATE": "[!! ]", "LOW": "[   ]"}

    SEP  = "=" * 72
    LINE = "-" * 72

    print("\n" + SEP)
    print("  {} | {}/{} | {} | quality={}".format(
        sent_id, company, year, report_type, quality))
    print(LINE)

    # Word-wrap text at 70 chars
    words = text.split()
    line, lines_out = "", []
    for w in words:
        if len(line) + len(w) + 1 > 70:
            lines_out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines_out.append(line)
    for l in lines_out:
        print("  " + l)
    print(LINE)

    # Tier 1
    print("  +-- TIER 1: SUGGESTED LABEL")
    print("  |   Label: [{}] {}    Confidence: {} ({})".format(
        label, LABEL_NAMES.get(label, "?"),
        conf, CONF_NAMES.get(conf, "")))
    print("  |")

    # Tier 2
    print("  +-- TIER 2: TOP 3 FEATURE TRIGGERS")
    print("  |   {:20s} {:6s}  {:30s}  {}".format(
        "Feature", "Score", "Signal", "Interpretation"))
    print("  |   " + "-" * 65)
    for f in features:
        sig = ", ".join(f.signals) if f.signals else "NONE"
        sig_trunc   = sig[:28]
        interp_trunc = f.interpretation[:35]
        print("  |   {:20s} {:6.2f}  {:30s}  {}".format(
            f.name, f.score, '"' + sig_trunc + '"', interp_trunc))
    print("  |")

    # Greenwashing risk
    gflag = GW_FLAGS.get(gw_label, "")
    print("  +-- [GREENWASHING RISK] {} {}  (score={:.2f})".format(
        gflag, gw_label, gw_risk))
    print(SEP)
