import json

with open("cost_data_metro.json", "r", encoding="utf-8") as f:
    data = json.load(f)

options = []
for cbsa, prof in data.items():
    label = prof.get("metro_name") or cbsa
    options.append({"value": cbsa, "label": label})

options.sort(key=lambda x: x["label"].lower())

with open("cbsa_dropdown.json", "w", encoding="utf-8") as f:
    json.dump(options, f, indent=2)

print(f"✅ Wrote {len(options)} options to cbsa_dropdown.json")
