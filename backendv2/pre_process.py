import re
import json
from datetime import datetime, timedelta

# ================= CONFIGURATION =================
# If the gap between messages is greater than this, start a new session
SESSION_TIME_THRESHOLD_HOURS = 3 
INPUT_FILE = "_chat.txt"
OUTPUT_FILE = "whatsapp_sessions_cleaned.json"

# List of exact phrases to exclude completely
EXCLUDED_PHRASES = [
    "<Media omitted>", 
    "Messages and calls are end-to-end encrypted",
    "Missed voice call",
    "Missed video call"
]
# =================================================

def parse_whatsapp_chat(filepath):
    # Regex to match the timestamp format from your sample: "11/11/24, 8:36 pm - Name:"
    # It handles the specific unicode space (\u202f) often found in WhatsApp exports
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})[\s\u202f]?([ap]m)\s-\s(.*?):\s(.*)$', re.IGNORECASE)
    
    sessions = []
    current_session = []
    last_datetime = None
    
    print(f"Reading {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        match = pattern.match(line)

        if match:
            date_str, time_str, am_pm, sender, message = match.groups()
            
            # 1. Clean Data: Skip if message is in exclusion list
            if message.strip() in EXCLUDED_PHRASES:
                continue

            # 2. Parse Date Time
            # Combine components to create a datetime object
            # Note: We replace the non-breaking space just in case for standard parsing
            full_date_str = f"{date_str} {time_str} {am_pm}".replace('\u202f', ' ')
            try:
                # Adjust format based on your file (dd/mm/yy vs mm/dd/yy)
                # Your sample "11/11/24" is ambiguous, assuming dd/mm/yy for India
                msg_datetime = datetime.strptime(full_date_str, "%d/%m/%y %I:%M %p")
            except ValueError:
                # Fallback if year is 4 digits
                msg_datetime = datetime.strptime(full_date_str, "%d/%m/%Y %I:%M %p")

            # 3. Session Logic
            if last_datetime:
                delta = msg_datetime - last_datetime
                # If gap is huge, push current session and start new
                if delta > timedelta(hours=SESSION_TIME_THRESHOLD_HOURS):
                    if current_session:
                        sessions.append(current_session)
                    current_session = []

            # 4. Add to current session
            current_session.append({
                "sender": sender,
                "timestamp": msg_datetime.isoformat(),
                "message": message
            })
            
            last_datetime = msg_datetime

        else:
            # Handle multi-line messages (newlines within a single message)
            if current_session:
                # Append this line to the previous message content
                current_session[-1]["message"] += f"\n{line}"

    # Append the very last session
    if current_session:
        sessions.append(current_session)

    return sessions

def save_sessions(sessions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)
    print(f"✅ Success! Processed {len(sessions)} distinct conversation sessions.")
    print(f"💾 Saved to {output_file}")

# Execution
if __name__ == "__main__":
    try:
        data = parse_whatsapp_chat(INPUT_FILE)
        save_sessions(data, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{INPUT_FILE}'. Make sure your export file is named correctly.")