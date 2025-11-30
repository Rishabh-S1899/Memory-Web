# generate_backstory_template_json.py
"""
Generates photos_backstories_template.json listing all images in PHOTO_FOLDER,
sorted strictly by filename (A–Z). NO chat data used.

You will manually edit the "backstory" field for each image.
"""

import os
import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

PHOTO_FOLDER = "data"
OUT_JSON = "photos_backstories_template.json"


def get_exif_date(fp):
    """Return EXIF DateTimeOriginal if present, else ''."""
    try:
        img = Image.open(fp)
        exif = img._getexif()
        if not exif:
            return ""
        for k, v in exif.items():
            name = TAGS.get(k, k)
            if name in ["DateTimeOriginal", "DateTime"]:
                # EXIF format: YYYY:MM:DD HH:MM:SS -> convert to YYYY-MM-DD HH:MM:SS
                return v.replace(":", "-", 2)
    except Exception:
        pass
    return ""


def main():
    folder = Path(PHOTO_FOLDER)
    if not folder.exists():
        print(f"Error: folder '{PHOTO_FOLDER}' does not exist.")
        return

    # Sort strictly by filename (alphabetical)
    image_files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    )

    if not image_files:
        print(f"No images found in {PHOTO_FOLDER}.")
        return

    entries = []
    for fname in image_files:
        fp = str(folder / fname)
        exif_date = get_exif_date(fp)

        entries.append({
            "id": fname,                    # simple stable ID = filename
            "filename": fname,              
            "filepath": os.path.abspath(fp),
            "exif_date": exif_date,
            "backstory": "WRITE YOUR BACKSTORY HERE",  # <-- write here!
            "tags": []
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"\nTemplate created: {OUT_JSON}")
    print(f"📸 Total images: {len(entries)}")
    print("👇 They are sorted alphabetically by filename for easy backstory writing.")


if __name__ == "__main__":
    main()
