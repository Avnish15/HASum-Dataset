import os
import json
import pandas as pd
from yt_dlp import YoutubeDL
from collections import defaultdict
from pathlib import Path

# Configuration
ANNOTATION_PATH = "activity_net.v1-3.min.json"
OUTPUT_DIR = "videos"
MIN_DURATION = 180  # seconds
MAX_DOWNLOADS_PER_CATEGORY = 20
PREFERRED_RESOLUTIONS = [720, 480]
COOKIES_FILE = "cookies.txt"

# Target categories
TARGET_CATEGORIES = ["Hand car wash", "Rafting", "Bike tricks", "Rock climbing", "Making sandwich", "Getting a haircut", "Grooming dog"]

# Load ActivityNet annotations
with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)
database = data["database"]

# Gather video info
category_to_videos = defaultdict(list)
for vid, info in database.items():
    duration = info.get("duration", 0)
    if duration < MIN_DURATION:
        continue
    url = info.get("url", "")
    for ann in info["annotations"]:
        label = ann["label"]
        if label in TARGET_CATEGORIES:
            category_to_videos[label].append((vid, url, duration))

# Filter categories with ≥20 videos
filtered_categories = {
    cat: sorted(vids, key=lambda x: x[2])
    for cat, vids in category_to_videos.items()
    if len(vids) >= MAX_DOWNLOADS_PER_CATEGORY
}

# YT-DLP options
def get_ydl_opts(save_dir, download=True):
    opts = {
        "quiet": True,
        "format": "mp4",
        "outtmpl": str(save_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "ignoreerrors": True,
        "retries": 3,
        "merge_output_format": "mp4",
        "cookiefile": COOKIES_FILE,
    }
    if download:
        opts["progress_hooks"] = [lambda d: print(f"✔ Downloaded: {d['filename']}") if d["status"] == "finished" else None]
    return opts

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Begin downloading
for category, videos in filtered_categories.items():
    save_dir = Path(OUTPUT_DIR) / category
    save_dir.mkdir(parents=True, exist_ok=True)

    excel_data = []
    downloaded = 0

    with YoutubeDL(get_ydl_opts(save_dir)) as ydl:
        for video_id, url, _ in videos:
            if downloaded >= MAX_DOWNLOADS_PER_CATEGORY:
                break

            mp4_path = save_dir / f"{video_id}.mp4"
            if mp4_path.exists():
                continue  # already downloaded

            print(f"🔍 Probing {video_id}...")

            try:
                with YoutubeDL(get_ydl_opts(save_dir, download=False)) as probe:
                    info = probe.extract_info(url, download=False)
                if info is None:
                    raise ValueError("No video info returned")
            except Exception as e:
                print(f"⚠ Skipping {video_id}: Metadata fetch failed ({e}).")
                continue

            # Check for valid resolution
            valid_format = any(
                f.get("ext") == "mp4" and f.get("height") in PREFERRED_RESOLUTIONS
                for f in info.get("formats", [])
            )
            if not valid_format:
                print(f"⚠ Skipping {video_id}: No valid resolution (480p or 720p).")
                continue

            try:
                print(f"⬇ Downloading {video_id} ({category})...")
                ydl.download([url])
                downloaded += 1

                # Save metadata
                excel_data.append({
                    "Video_ID": video_id,
                    "Video_URL": url,
                    "Video_title": info.get("title", ""),
                    "Channel_Name": info.get("uploader", ""),
                    "Video_Description": info.get("description", ""),
                    "Video_Tags": category
                })

            except Exception as e:
                print(f"❌ Failed to download {video_id}: {e}")

    # Save metadata to Excel
    if excel_data:
        df = pd.DataFrame(excel_data)
        excel_path = save_dir / f"{category}_metadata.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"📄 Metadata saved to {excel_path}")
