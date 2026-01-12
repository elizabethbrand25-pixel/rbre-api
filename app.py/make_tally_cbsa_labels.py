import json

INPUT_FILE = "cost_data_metro.json"
OUTPUT_FILE = "tally_cbsa_labels.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

labels = []

for cbsa, prof in data.items():
    name = prof.get("metro_name")
    if isinstance(name, str) and name.strip():
        labels.append(name.strip())

labels = sorted(set(labels), key=lambda x: x.lower())

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for label in labels:
        f.write(label + "\n")

print(f"✅ Wrote {len(labels)} Tally-ready labels to {OUTPUT_FILE}")
