#!/usr/bin/env python3
# extract_memories_final.py
"""
Final robust extraction script (drop-in).
- Parses WhatsApp _chat.txt
- Sessionizes by TIME_GAP_HOURS
- Sliding-window chunking
- Calls local LLM (Ollama / mistral-nemo) for JSON extraction (robust to returned types)
- Handles both `format="json"` (object) and string outputs
- Logs parsing issues and saves debug samples
- Deduplicates and writes memories_gold.json

Usage:
  - Put your WhatsApp export at INPUT_FILE (default "_chat.txt")
  - Ensure Ollama + mistral-nemo accessible via langchain_ollama.ChatOllama
  - Run: python extract_memories_final.py
"""

import re
import json
import hashlib
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import os

# --- CONFIGURATION ---
INPUT_FILE = "_chat.txt"
OUTPUT_FILE = "memories_gold.json"
DEBUG_SAVE = "debug_responses.json"

# Logic Settings
TIME_GAP_HOURS = 2       # Split into new session if silence > 2 hours
CHUNK_SIZE = 75          # Messages per LLM call
OVERLAP = 20             # Overlap to catch context at boundaries

# Model Setup
# If you previously relied on format="json", you can keep it; script handles both.
try:
    from langchain_ollama import ChatOllama
except Exception as e:
    print("ERROR: langchain_ollama not available. Install or adjust your LLM wrapper.")
    raise

# Keep temperature low for deterministic results; set format if you like.
llm = ChatOllama(model="mistral-nemo", temperature=0.1, format="json")

# --- PROMPT ---
extraction_prompt = """
You are a Relationship Historian.
Analyze this chat log (Hinglish/English).

Extract "Core Memories" into a JSON list.
A Core Memory is:
- Romantic (Confessions, compliments, "I love you")
- Milestones (Dates, Trips, Anniversaries, Gifts)
- Funny (Inside jokes, funny roasts)
- Future Promises ("We will go to Paris")

IGNORE:
- Logistics (Traffic, Food delivery, Location sharing)
- Small talk (Gm, Gn, Ok, Hmm)
- System messages

Output Format (strict JSON array):
[
  {
    "memory": "Detailed summary of the moment",
    "date": "DD/MM/YY (The date found in the log)",
    "category": "romantic" | "funny" | "milestone" | "trip"
  }
]
If nothing relevant, return [].
**IMPORTANT**: Output only the JSON array (no extra commentary). If you cannot output exactly JSON array,
you may still return a JSON object containing a list under a top-level key (e.g., {"memories": [...]}) — the parser will handle that.
"""

# --------------------
# 1. Parsing & Sessionizing
# --------------------
print("⏳ Parsing Chat Log...")

def parse_line(line):
    # Regex: [24/01/25, 13:00] Name: Msg  (adapt if your export format differs)
    match = re.search(r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]m)?)\]?\s-?\s?(.*)', line, re.IGNORECASE)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        text = match.group(3).strip()
        # Try to parse datetime; support multiple formats
        dt_obj = None
        dt_str = f"{date_str} {time_str}"
        for fmt in ["%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M", "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p"]:
            try:
                dt_obj = datetime.strptime(dt_str, fmt)
                break
            except ValueError:
                continue
        # Fallback: try removing seconds or AM/PM
        if dt_obj is None:
            try:
                dt_obj = datetime.strptime(dt_str, "%d/%m/%y %H:%M")
            except Exception:
                return None
        return {"dt": dt_obj, "str": f"[{date_str} {time_str}] {text}", "date_only": date_str}
    return None

all_messages = []
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: Input file {INPUT_FILE} not found. Put your WhatsApp export at this path.")
    sys.exit(1)

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parsed = parse_line(line)
        if parsed:
            all_messages.append(parsed)

print(f"📄 Parsed {len(all_messages)} valid messages.")

# Group into Sessions based on TIME_GAP
sessions = []
current_session = []
last_dt = None

for msg in all_messages:
    if last_dt and (msg['dt'] - last_dt) > timedelta(hours=TIME_GAP_HOURS):
        if current_session:
            sessions.append(current_session)
        current_session = []
    current_session.append(msg['str'])
    last_dt = msg['dt']

if current_session:
    sessions.append(current_session)

print(f"🕒 Organized into {len(sessions)} distinct conversations (Sessions).")

# --------------------
# 2. Chunking (Sliding Window)
# --------------------
final_chunks = []

for session in sessions:
    if len(session) <= CHUNK_SIZE:
        final_chunks.append("\n".join(session))
    else:
        start = 0
        while start < len(session):
            end = start + CHUNK_SIZE
            window = session[start:end]
            final_chunks.append("\n".join(window))
            if end >= len(session):
                break
            start += (CHUNK_SIZE - OVERLAP)

print(f"🧩 Ready to process {len(final_chunks)} chunks with LLM.")

# --------------------
# 3. LLM extraction (robust)
# --------------------
raw_memories = []
debug_list = []
MAX_DEBUG_SAVE = 30
MAX_PRINT = 8
bad_chunks = 0
total_parsed_items = 0

print("🚀 Starting Extraction (robust mode)...")
for i, chunk in enumerate(tqdm(final_chunks)):
    try:
        response = llm.invoke([
            ("system", extraction_prompt),
            ("human", chunk)
        ])
        content = response.content

        # Save debug sample (first MAX_DEBUG_SAVE)
        if len(debug_list) < MAX_DEBUG_SAVE:
            # store small repr
            try:
                content_repr = repr(content)
            except Exception:
                content_repr = str(type(content))
            debug_list.append({
                "chunk_index": i,
                "content_type": str(type(content)),
                "content_repr": content_repr[:2000]
            })

        # Print first few items to console for quick inspection
        if i < MAX_PRINT:
            print(f"\n--- DEBUG response #{i} ---")
            print("TYPE:", type(content))
            print("REPR:", (repr(content)[:1000] if isinstance(content, (str, list, dict)) else str(content)) )

        parsed_any = False

        # Case 1: LLM returned parsed Python object (list)
        if isinstance(content, list):
            raw_memories.extend(content)
            total_parsed_items += len(content)
            parsed_any = True

        # Case 2: content is dict possibly containing a list
        elif isinstance(content, dict):
            found = False
            # common keys: memories, data, items, result
            for k in ("memories", "data", "items", "result", "results"):
                if k in content and isinstance(content[k], list):
                    raw_memories.extend(content[k])
                    total_parsed_items += len(content[k])
                    parsed_any = True
                    found = True
                    break
            if not found:
                # attempt to find any list value
                for k, v in content.items():
                    if isinstance(v, list):
                        raw_memories.extend(v)
                        total_parsed_items += len(v)
                        parsed_any = True
                        break

        # Case 3: string -> try extract JSON safely
        elif isinstance(content, str):
            s = content.strip()
            if s in ("[]", "null", "None", ""):
                # nothing returned
                parsed_any = False
            else:
                # strip code fences/backticks if present
                if s.startswith("```") and s.endswith("```"):
                    s = s.strip("` \n")
                # try to parse entire string as JSON first
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        raw_memories.extend(obj)
                        total_parsed_items += len(obj)
                        parsed_any = True
                    elif isinstance(obj, dict):
                        # try common list keys
                        for k in ("memories", "data", "items", "result", "results"):
                            if k in obj and isinstance(obj[k], list):
                                raw_memories.extend(obj[k])
                                total_parsed_items += len(obj[k])
                                parsed_any = True
                                break
                        if not parsed_any:
                            # find first list value in dict
                            for _, v in obj.items():
                                if isinstance(v, list):
                                    raw_memories.extend(v)
                                    total_parsed_items += len(v)
                                    parsed_any = True
                                    break
                except Exception:
                    # fallback: try to slice out the first JSON array substring
                    if "[" in s and "]" in s:
                        try:
                            start = s.find("[")
                            end = s.rfind("]") + 1
                            json_str = s[start:end]
                            extracted = json.loads(json_str)
                            if isinstance(extracted, list):
                                raw_memories.extend(extracted)
                                total_parsed_items += len(extracted)
                                parsed_any = True
                        except Exception as e:
                            # parsing failed for this chunk
                            # print a short message for debugging but continue
                            if i < MAX_PRINT:
                                print(f"Warning parsing chunk {i}: {e}")
                                print("snippet:", s[:300])
                            parsed_any = False

        if not parsed_any:
            bad_chunks += 1

    except Exception as e:
        print(f"ERROR invoking LLM on chunk {i}: {e}")
        traceback.print_exc()
        bad_chunks += 1
        continue

# Save debug samples for inspection
with open(DEBUG_SAVE, "w", encoding="utf-8") as f:
    json.dump(debug_list, f, indent=2)

print(f"\nExtraction finished. Total parsed memory items collected: {len(raw_memories)} (total_parsed_items var: {total_parsed_items})")
print(f"Bad / unparsed chunks: {bad_chunks} out of {len(final_chunks)}")
print(f"Saved debug samples to {DEBUG_SAVE}")

# --------------------
# 4. Deduplication & normalization
# --------------------
print("🧹 Deduplicating overlaps and normalizing...")

def canonicalize_memory_item(item):
    """
    Ensure memory is a dict with keys: memory, date, category.
    Attempt best-effort normalization for common cases.
    """
    if isinstance(item, str):
        return {"memory": item, "date": "", "category": ""}
    if not isinstance(item, dict):
        return {"memory": str(item), "date": "", "category": ""}

    mem_text = item.get("memory") or item.get("memory_text") or item.get("text") or ""
    date_val = item.get("date") or item.get("date_found") or ""
    cat = item.get("category") or item.get("tag") or item.get("type") or ""
    # simple cleanup
    mem_text = mem_text.strip() if isinstance(mem_text, str) else str(mem_text)
    date_val = date_val.strip() if isinstance(date_val, str) else str(date_val)
    cat = cat.strip().lower() if isinstance(cat, str) else ""
    return {"memory": mem_text, "date": date_val, "category": cat}

unique_memories = []
seen_hashes = set()
for item in raw_memories:
    canon = canonicalize_memory_item(item)
    fingerprint = f"{canon.get('date','')}-{canon.get('memory','')}".lower().strip()
    h = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_memories.append(canon)

# Optional: sort by date if parsable (most recent first)
def parse_possible_date(dstr):
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(dstr, fmt)
        except Exception:
            continue
    return None

# Keep original order but you may sort; here we keep insertion order (first found)
print(f"✅ DONE! Saved {len(unique_memories)} cleaned memories to {OUTPUT_FILE}")

# --------------------
# 5. Save final JSON
# --------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(unique_memories, f, indent=2, ensure_ascii=False)

print("All finished.")
