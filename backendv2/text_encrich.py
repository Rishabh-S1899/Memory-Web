import json
import ollama
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_FILE = "whatsapp_sessions_cleaned.json"
OUTPUT_FILE = "gemma_knowledge_base_enriched_final.json"

# THE UPGRADE: Qwen 2.5 14B (Fits in 10GB VRAM)
MODEL_NAME = "llama3.1:8b" 

USER_NAME = "Rishabh Shukla" 
PARTNER_NAME = "Chetna 🌻"

MAX_CHUNK_CHARS = 12000 
OVERLAP_MESSAGES = 5 
# =================================================

SYSTEM_PROMPT = f"""
You are an expert Relationship Archivist. 
The conversation is between '{USER_NAME}' (User) and '{PARTNER_NAME}' (Partner).
The text is in "Hinglish".

You MUST output a valid JSON object. Do not add any text before or after the JSON.

Required JSON Structure:
{{
  "summary_english": "Detailed summary of the event. Be specific.",
  "significance_score": 8, 
  "memory_category": "Logistics, Casual, Deep, Event, Conflict, Romance",
  "keywords": ["List", "of", "5", "keywords"],
  "mood": "Emotion",
  "notable_quotes": ["Quote 1", "Quote 2"]
}}

SCORING:
- 1-3: Logistics/Boring
- 4-10: Meaningful
"""

def split_session_safely(session_data):
    chunks = []
    current_chunk = []
    current_char_count = 0
    
    for i, msg in enumerate(session_data):
        msg_len = len(msg['sender']) + len(msg['message']) + 5
        
        if current_char_count + msg_len > MAX_CHUNK_CHARS:
            if current_chunk:
                chunks.append(current_chunk)
            start_index = max(0, i - OVERLAP_MESSAGES)
            overlap_data = session_data[start_index : i]
            current_chunk = overlap_data + [msg]
            current_char_count = sum(len(m['sender']) + len(m['message']) + 5 for m in current_chunk)
        else:
            current_chunk.append(msg)
            current_char_count += msg_len
            
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def process_chunk(chunk_data, date_ref, time_ref, chunk_index, total_chunks):
    chat_text = ""
    for msg in chunk_data:
        chat_text += f"{msg['sender']}: {msg['message']}\n"

    try:
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Analyze this chat part ({chunk_index}/{total_chunks}):\n\n{chat_text}"},
            ],
            options={'num_ctx': 8192}, 
            format='json'
        )
        
        content = response['message']['content']
        analysis = json.loads(content)
        
        # Quality Control
        if 'keywords' not in analysis or not isinstance(analysis['keywords'], list):
            analysis['keywords'] = []
        if 'significance_score' not in analysis:
            analysis['significance_score'] = 5
        if 'mood' not in analysis:
            analysis['mood'] = "Neutral"

        # === ID FIX IS HERE ===
        # We include the TIME (HHMM) in the ID to make it unique per session
        unique_id = f"{date_ref}_{time_ref}_{chunk_index}"

        return {
            "original_chat": chunk_data, 
            "summary_vector_source": analysis.get('summary_english', 'No summary generated'), 
            "metadata": {
                "date": date_ref, 
                "chunk_id": unique_id, # <--- NEW UNIQUE ID
                "score": analysis['significance_score'],
                "category": analysis.get('memory_category', 'General'),
                "mood": analysis['mood'],
                "keywords": analysis['keywords'],
                "quotes": analysis.get('notable_quotes', [])
            }
        }
    except Exception as e:
        print(f"⚠️ Error parsing chunk: {e}")
        return None

def main():
    print(f"🚀 Starting Phase 2 with {MODEL_NAME}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    enriched_data = []
    
    print(f"Processing {len(sessions)} raw sessions...")
    
    for session in tqdm(sessions):
        if not session: continue
        
        # 1. Extract Date AND Time for uniqueness
        # Format: "2024-11-12T11:38:00"
        timestamp = session[0]['timestamp']
        date_ref = timestamp[:10] 
        # Extract HHMM (e.g., 1138) to ensure morning/evening sessions don't clash
        try:
            time_ref = timestamp[11:16].replace(':', '') 
        except:
            time_ref = "0000"

        # 2. Split
        chunks = split_session_safely(session)
        
        # 3. Process
        for i, chunk in enumerate(chunks):
            # Pass time_ref to the processor
            result = process_chunk(chunk, date_ref, time_ref, i+1, len(chunks))
            
            if result:
                if result['metadata']['score'] >= 4:
                    enriched_data.append(result)

    print(f"✅ Processing Complete. Saved {len(enriched_data)} memories.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

# import json
# import ollama
# from tqdm import tqdm
# import time

# # ================= CONFIGURATION =================
# INPUT_FILE = "whatsapp_sessions_cleaned.json"
# OUTPUT_FILE = "knowledge_base_enriched_final.json" # New output file name
# MODEL_NAME = "llama3.1:8b" 

# # UPDATE THESE TO MATCH YOUR WHATSAPP EXPORT NAMES EXACTLY
# USER_NAME = "Rishabh Shukla" 
# PARTNER_NAME = "Chetna 🌻"

# # Max characters per chunk (approx 3.5 chars = 1 token). 
# MAX_CHUNK_CHARS = 12000 
# OVERLAP_MESSAGES = 5 
# # =================================================

# SYSTEM_PROMPT = f"""
# You are an expert Relationship Archivist. 
# The conversation is between '{USER_NAME}' (User) and '{PARTNER_NAME}' (Partner).
# The text is in "Hinglish".

# You MUST output a valid JSON object. Do not add any text before or after the JSON.

# Required JSON Structure:
# {{
#   "summary_english": "Detailed summary of the event. Be specific. Example: '{USER_NAME} teased {PARTNER_NAME} about her driving skills'.",
#   "significance_score": 8, 
#   "memory_category": "One of: Logistics, Casual, Deep, Event, Conflict, Romance",
#   "keywords": ["List", "of", "5", "specific", "keywords"],
#   "mood": "One word emotion",
#   "notable_quotes": ["Extract 1-2 specific messages in original Hinglish"]
# }}

# SCORING GUIDE:
# - 1-3: Logistics/Boring (e.g. "Reached?", "Ok")
# - 7-10: Significant (Inside jokes, fights, trips, deep talks)
# """

# def split_session_safely(session_data):
#     """Splits massive sessions into overlapping chunks."""
#     chunks = []
#     current_chunk = []
#     current_char_count = 0
    
#     for i, msg in enumerate(session_data):
#         msg_len = len(msg['sender']) + len(msg['message']) + 5
        
#         if current_char_count + msg_len > MAX_CHUNK_CHARS:
#             if current_chunk:
#                 chunks.append(current_chunk)
            
#             # Create Overlap
#             start_index = max(0, i - OVERLAP_MESSAGES)
#             overlap_data = session_data[start_index : i]
#             current_chunk = overlap_data + [msg]
#             current_char_count = sum(len(m['sender']) + len(m['message']) + 5 for m in current_chunk)
#         else:
#             current_chunk.append(msg)
#             current_char_count += msg_len
            
#     if current_chunk:
#         chunks.append(current_chunk)
#     return chunks

# def process_chunk(chunk_data, date_ref, chunk_index, total_chunks):
#     chat_text = ""
#     for msg in chunk_data:
#         chat_text += f"{msg['sender']}: {msg['message']}\n"

#     try:
#         # We use format='json' to force structured output
#         response = ollama.chat(
#             model=MODEL_NAME, 
#             messages=[
#                 {'role': 'system', 'content': SYSTEM_PROMPT},
#                 {'role': 'user', 'content': f"Analyze this chat part ({chunk_index}/{total_chunks}):\n\n{chat_text}"},
#             ],
#             options={'num_ctx': 8192}, # Force large context
#             format='json' # <--- THE KEY UPGRADE
#         )
        
#         content = response['message']['content']
#         analysis = json.loads(content)
        
#         # === QUALITY CONTROL LAYER ===
#         # If the LLM forgets a field, we fill it with a placeholder so the script never crashes
        
#         # 1. Ensure Keywords exist
#         if 'keywords' not in analysis or not isinstance(analysis['keywords'], list):
#             # Fallback: Use words from the summary if keywords are missing
#             analysis['keywords'] = analysis.get('summary_english', '').split()[:5]
            
#         # 2. Ensure Score exists
#         if 'significance_score' not in analysis:
#             analysis['significance_score'] = 5
            
#         # 3. Ensure Mood exists
#         if 'mood' not in analysis:
#             analysis['mood'] = "Neutral"

#         return {
#             "original_chat": chunk_data, 
#             "summary_vector_source": analysis.get('summary_english', 'No summary generated'), 
#             "metadata": {
#                 "date": date_ref, 
#                 "chunk_id": f"{date_ref}_{chunk_index}",
#                 "score": analysis['significance_score'],
#                 "category": analysis.get('memory_category', 'General'),
#                 "mood": analysis['mood'],
#                 "keywords": analysis['keywords'], # Now guaranteed to exist
#                 "quotes": analysis.get('notable_quotes', [])
#             }
#         }
#     except Exception as e:
#         print(f"⚠️ Error parsing chunk {chunk_index} of {date_ref}: {e}")
#         return None

# def main():
#     print("🚀 Starting Phase 2: Enrichment (v4 - High Quality Mode)")
    
#     try:
#         with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#             sessions = json.load(f)
#     except FileNotFoundError:
#         print(f"❌ Error: {INPUT_FILE} not found. Run Step 1 first.")
#         return

#     enriched_data = []
    
#     print(f"Processing {len(sessions)} raw sessions...")
    
#     for session in tqdm(sessions):
#         if not session: continue
        
#         # Extract date safely
#         timestamp = session[0]['timestamp']
#         date_ref = timestamp[:10] if isinstance(timestamp, str) else "Unknown_Date"
        
#         # 1. Split
#         chunks = split_session_safely(session)
        
#         # 2. Process
#         for i, chunk in enumerate(chunks):
#             result = process_chunk(chunk, date_ref, i+1, len(chunks))
            
#             if result:
#                 # Filter noise (Keep score 4+)
#                 if result['metadata']['score'] >= 4:
#                     enriched_data.append(result)

#     print(f"✅ Processing Complete. Saved {len(enriched_data)} high-quality memories.")
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(enriched_data, f, indent=2, ensure_ascii=False)

# if __name__ == "__main__":
#     main()