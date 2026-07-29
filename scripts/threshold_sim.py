import csv, sys
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

rows = list(csv.DictReader(open('annotations/full_corpus_labeled.csv', encoding='utf-8')))
model_rows = [r for r in rows if r['source'] == 'model']

total = len(model_rows)
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

print("Total model-predicted sentences: {}".format(total))
print()
print("{:<12} {:<14} {:<14} {:<10} {:<10}".format(
    "Threshold", "High-conf", "Flagged-QA", "V(high)", "S(high)"))
print("-" * 60)

for t in thresholds:
    high = [r for r in model_rows if float(r['confidence']) >= t]
    low  = [r for r in model_rows if float(r['confidence']) <  t]
    v_h  = sum(1 for r in high if r['label'] == 'V')
    s_h  = sum(1 for r in high if r['label'] == 'S')
    marker = " <-- current" if t == 0.65 else (" <-- proposed" if t == 0.60 else "")
    print("{:<12} {:<14} {:<14} {:<10} {:<10}{}".format(
        t, len(high), len(low), v_h, s_h, marker))
