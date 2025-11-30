# remove_media_lines.py
"""
Preprocess WhatsApp export to remove lines containing "media omitted".
Creates a backup: INPUT_FILE + ".bak"
Writes cleaned output to the same INPUT_FILE (overwrites).
"""

import re
import shutil
from pathlib import Path

INPUT_FILE = "_chat.txt"
BACKUP_SUFFIX = ".bak"
PATTERN = re.compile(r"(?:<|\[)?\s*media\s+omitted\s*(?:>|\])?", re.IGNORECASE)  # matches <Media omitted>, [Media omitted], Media omitted

def remove_media_lines(input_path: str):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    # backup_path = p.with_suffix(p.suffix + BACKUP_SUFFIX) if p.suffix else Path(str(p) + BACKUP_SUFFIX)
    # # Make a safe backup (do not overwrite existing .bak)
    # if not backup_path.exists():
    #     shutil.copy2(p, backup_path)
    #     print(f"Backup created at: {backup_path}")
    # else:
    #     print(f"Backup already exists at: {backup_path}")

    kept_lines = []
    removed = 0
    total = 0

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            # quick check first to avoid regex cost on common lines
            low = line.lower()
            if "media omitted" in low:
                # use regex to be conservative about variants
                if PATTERN.search(line):
                    removed += 1
                    continue
            kept_lines.append(line)

    # overwrite original with cleaned lines
    with p.open("w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    print(f"Processed {total} lines. Removed {removed} lines containing 'media omitted'.")
    print(f"Cleaned file written to: {p} ")

if __name__ == "__main__":
    remove_media_lines(INPUT_FILE)
