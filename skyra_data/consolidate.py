import os
import json

base_dir = "parsed_frames"
master_data = []

print(" Crawling through folders to find annotations...")

for root, dirs, files in os.walk(base_dir):
    if "timestamps.txt" in files:
        txt_path = os.path.join(root, "timestamps.txt")
        video_id = os.path.basename(root)
        try:
            with open(txt_path, "r") as f:
                content = f.read().strip()
                if content:
                    master_data.append({
                        "video_id": video_id,
                        "cot_response": content
                    })
        except Exception as e:
            continue

print(f" Found {len(master_data)} annotations!")
with open("skyra_master.json", "w") as f:
    json.dump(master_data, f)
print(" Created skyra_master.json")
