# Phase 3 Annotation Guidelines
## GreenWashDet — Environmental Sentence Labeling Schema

**Version**: 1.0
**Corpus**: 1,172 environmental sentences from BRR/BRSR reports
**Companies**: JSW Steel, Tata Steel, UltraTech Cement (2019-2026)
**Target IAA**: Cohen's Kappa >= 0.80 before batch annotation

---

## 1. The Three Labels

| Label | Name              | One-line definition                                      |
|-------|-------------------|----------------------------------------------------------|
| V     | Vague             | Aspirational/marketing language, no measurable specifics |
| S     | Substantive       | Specific, quantified, or verifiable environmental claim  |
| N     | Numeric-financial | Raw data row / table entry with minimal prose context    |
| ?     | Unsure            | Cannot classify — flag for team discussion               |

---

## 2. Label Definitions and Examples

### V — VAGUE

A sentence that makes an environmental claim using **hedging language, general
commitments, or aspirational framing** WITHOUT any measurable data (no numbers,
no dates, no targets, no named technologies, no named locations).

Signal words (lean toward V):
  aim, strive, seek, endeavour, intend, aspire, promote, focus on,
  work towards, committed to, dedicated to, significant, meaningful

**Decision rule**: If you stripped all words and kept only numbers and named
entities, would NOTHING meaningful remain? -> V

Real examples from corpus:
  V: We are committed to reducing our environmental impact across all operations.
  V: The Company promotes sustainable practices and aims to minimise its carbon footprint.
  V: We strive to achieve net-zero in the long term through responsible manufacturing.
  V: Our focus remains on reducing emissions and conserving natural resources.


### S — SUBSTANTIVE

A sentence that includes **at least ONE verifiable specific**: a number,
percentage, year, named technology, geographic location, or past-tense action.

Signal elements (lean toward S):
  - Quantifiers: percentages, tonnes, GJ, MW, kL, m3, tCO2/tcs
  - Time references: by 2030, in FY2023, since 2019, over five years
  - Named technologies: CCUS, waste heat recovery, solar PV, blast furnace injection
  - Named locations: Vijayanagar plant, Dolvi Works, Jamshedpur
  - Past-tense verbs: reduced, achieved, commissioned, installed, deployed

**Decision rule**: Could an ESG analyst verify or falsify this claim? -> S

Real examples from corpus:
  S: We reduced specific CO2 emissions by 15% in FY2023 vs FY2019 baseline.
  S: 50 MW solar commissioned at Vijayanagar plant in FY2022-23.
  S: GHG intensity declined from 2.45 to 2.31 tCO2e/tcs in FY23.
  S: Company recycled 89% of solid waste at Vijayanagar facility in FY2019.


### N — NUMERIC-FINANCIAL

A **raw data entry** from a table — typically NO full grammatical sentence
structure, just measurements, units, and labels side-by-side.

Signal elements (lean toward N):
  - No main verb (or only copula is/are)
  - Mostly numbers with unit abbreviations: tCO2e, GJ, MWh, kL, m3, MnT
  - Multiple time periods listed in parallel
  - Column-like formatting artifacts from PDF extraction

**Decision rule**: Is this a data row or a sentence? -> N if data row

Real examples from corpus:
  N: Total Scope 1 emissions: 1,23,45,678 tCO2e  FY24: 2.31  FY23: 2.45
  N: Water consumption 3.8 m3/tcs (FY24), 4.1 m3/tcs (FY23)
  N: Absolute emissions Scope 1 MnT  66.0  75.7  70.2


---

## 3. Decision Tree

    Does the sentence have subject + verb + object (full sentence)?
      NO  -> N (data row)
      YES -> Does it contain specific data (numbers, dates, targets, named tech)?
               YES -> S  (verify: is it fact-checkable?)
               NO  -> Are hedging verbs present (aim, strive, commit)?
                        YES -> V
                        NO  -> V (general statement)


---

## 4. Edge Cases and Rules

Rule 1: Quantified targets (future tense is OK for S)
  GOOD: "We target a 30% reduction in water intensity by 2030"  -> S
  BAD:  "We aim to improve our environmental performance"        -> V

Rule 2: Past tense without numbers
  "Over the past decade we have invested in renewable energy"         -> V (no numbers)
  "We invested Rs 1,200 crores in FY2023, reducing Scope 1 by 8%"   -> S

Rule 3: Multi-part sentences - label the dominant part
  "We are committed to sustainability and reduced GHG by 12% since 2019" -> S

Rule 4: BRSR template prompts (regulatory format questions)
  "Provide details of greenhouse gas emissions (Scope 1 and Scope 2)..." -> ? (skip)

Rule 5: Product claims with names
  "Carbon-neutral cement product launched in FY23"   -> S (named product + year)
  "We produce environmentally friendly products"     -> V

Rule 6: N vs S for sentences with numbers
  "3.8 m3/tcs (FY24), 4.1 m3/tcs (FY23)"   -> N (no verb, pure data)
  "Water intensity improved to 3.8 m3/tcs in FY24 from 4.1 in FY23" -> S


---

## 5. Confidence Levels

  1 = Very confident (clear-cut case matching schema rules above)
  2 = Moderately confident (some ambiguity, label is defensible)
  3 = Low confidence (forced a label; almost marked ?)

IMPORTANT: When BOTH raters give confidence=1 but DISAGREE, that sentence
is a priority revision case for the schema.


---

## 6. IAA Workflow

**Step 1**: Each rater annotates Batch 1 (100 sentences) INDEPENDENTLY
  python scripts/annotation_tool.py --rater rater1 --batch batch1 --n 100
  python scripts/annotation_tool.py --rater rater2 --batch batch1 --n 100

**Step 2**: Compute Cohen's Kappa
  python scripts/kappa_calculator.py --r1 rater1 --r2 rater2 --batch batch1

**Step 3**: Interpret kappa
  kappa >= 0.80  -> PASS: proceed to full corpus annotation
  0.60 to 0.79  -> MODERATE: review disagreements CSV, update guidelines, re-annotate
  < 0.60        -> FAIL: mandatory schema revision session required

**Step 4** (if PASS): Batch annotate remainder
  python scripts/annotation_tool.py --rater rater1 --batch batch2 --n 572
  python scripts/annotation_tool.py --rater rater1 --batch batch3 --n 500

**Step 5**: 10% spot-check by second rater for quality assurance


---

## 7. Sector-Specific Vocabulary (Steel / Cement)

  "net-zero pathway" (no date or %)           -> V
  "net-zero by 2050 with 30% by 2030"         -> S
  "carbon-neutral cement" (named product)     -> S
  "CCUS pilot at Jamshedpur"                  -> S (named tech + location)
  "promoting green chemistry"                 -> V
  "kiln efficiency improved by 3%"            -> S
  "BRR regulatory template question text"     -> ? (skip)
  Row of numbers with GJ or MWh units only    -> N


---

## 8. Annotation Tool Quick Reference

  Start annotating:
    python scripts/annotation_tool.py --rater rater1 --batch batch1

  Keys during annotation:
    V / S / N / ?   assign label
    H               display schema reminder
    B               undo last annotation
    Q               save and quit

  Output files:
    annotations/rater1_batch1.csv         <- your labels
    annotations/disagreements_batch1.csv  <- after kappa calculation
    logs/phase3_kappa.txt                 <- kappa results
