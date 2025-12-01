# import json
# import ollama
# import os

# # ================= CONFIGURATION =================
# INPUT_FILE = "labelled_photos.json"
# OUTPUT_FILE = "images_enriched_final.json"
# MODEL_NAME = "llama3.1:8b" 

# # Identity Variables
# USER_NAME = "Rishabh"
# PARTNER_NAME = "Chetna"
# # =================================================

# def load_data(filepath):
#     if not os.path.exists(filepath):
#         print(f"❌ Error: {filepath} not found.")
#         return []
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def save_data(data, filepath):
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     print(f"💾 Saved.")

# def generate_questions(backstory):
#     """Asks Llama to generate probing questions based on the shallow backstory."""
#     system_prompt = f"""
#     You are a helpful friend helping {USER_NAME} remember details about a photo with his girlfriend {PARTNER_NAME}.
    
#     Read the short image description provided.
#     Generate 1 to 3 short, casual follow-up questions to help extract the "Vibe".
    
#     Focus on:
#     - Specific emotions (Who was happy? Who was annoyed?)
#     - Context (Was this a date? A reunion? A random meetup?)
    
#     Output ONLY the questions, separated by newlines.
#     """
    
#     try:
#         response = ollama.chat(
#             model=MODEL_NAME,
#             messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': f"Backstory: {backstory}"}
#             ]
#         )
#         return response['message']['content'].strip()
#     except Exception as e:
#         return f"How did {USER_NAME} and {PARTNER_NAME} feel?\nWhat was the occasion?"

# def synthesize_memory(old_backstory, user_notes):
#     """Combines old facts + new user notes into a rich, vector-friendly paragraph."""
#     system_prompt = f"""
#     You are a strict Data Archivist. 
#     Merge the 'Original Description' and 'User Notes' into a concise context block.
    
#     CRITICAL RULES:
#     1. STRICTLY NO FLOWERY LANGUAGE. (No words like "bustling", "palpable", "radiating", "invigorated").
#     2. Do NOT invent actions or feelings not explicitly mentioned.
#     3. Use the names '{USER_NAME}' and '{PARTNER_NAME}'.
#     4. Format: A direct factual description followed by the mood. Max 3 sentences.
    
#     Example Output:
#     "{USER_NAME} and {PARTNER_NAME} were at BHEL Stadium in Haridwar for a workout. They were both sweaty after the session but felt chill and relaxed."
    
#     Output strictly valid JSON:
#     {{
#       "rich_backstory": "The concise description...",
#       "tags": ["tag1", "tag2", "tag3", "tag4"]
#     }}
#     """
    
#     prompt_content = f"Original Description: {old_backstory}\nUser Notes: {user_notes}"
    
#     try:
#         response = ollama.chat(
#             model=MODEL_NAME,
#             messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': prompt_content}
#             ],
#             format='json' 
#         )
#         return json.loads(response['message']['content'])
#     except Exception as e:
#         print(f"⚠️ Error generating new backstory: {e}")
#         return None

# def main():
#     images = load_data(INPUT_FILE)
#     if not images: return

#     print(f"📸 Loaded {len(images)} images.\n")
#     print(f"💡 TIP: When answering, mention names! E.g. '{PARTNER_NAME} was happy' instead of just 'Happy'.\n")

#     for i, img in enumerate(images):
#         # Skip if already enriched
#         if len(img.get('tags', [])) > 0:
#             continue

#         print(f"--------------------------------------------------")
#         print(f"Image {i+1}/{len(images)}: {img['filename']}")
#         print(f"🖼️  Current: \"{img['backstory']}\"")
        
#         # 1. AI asks questions
#         print("\n🤔 AI is thinking of questions...")
#         questions = generate_questions(img['backstory'])
#         print(f"\n{questions}\n")
        
#         # 2. User answers
#         print(f"(Type your answer describing the moment for {USER_NAME} & {PARTNER_NAME})")
#         user_input = input("📝 Answer: ")
        
#         if user_input.lower() in ['exit', 'quit']:
#             break
#         if user_input.lower() == 'skip':
#             continue

#         # 3. Synthesize
#         print("\n✨ Writing new memory...")
#         result = synthesize_memory(img['backstory'], user_input)
        
#         if result:
#             new_story = result['rich_backstory']
#             new_tags = result['tags']
            
#             print(f"✅ Final: {new_story}")
#             print(f"🏷️  Tags: {new_tags}\n")
            
#             img['backstory'] = new_story
#             img['tags'] = new_tags
            
#             save_data(images, OUTPUT_FILE)
#         else:
#             print("❌ Failed to process. Skipping.")

#     print("\n🎉 Done! Now run Step 3 (Build DB).")

# if __name__ == "__main__":
#     main()



import json
import ollama
from tqdm import tqdm
import os

# ================= CONFIGURATION =================
INPUT_FILE = "labelled_photos.json"
OUTPUT_FILE = "images_tagged.json" # We save to a new file to be safe
MODEL_NAME = "gemma3:12b" # Or "llama3.1", whichever is loaded
# =================================================

def generate_tags(backstory):
    system_prompt = """
    You are an AI Tagger for a personal photo album.
    Read the provided backstory.
    Extract 5-10 specific, searchable keywords (Entities, Locations, Activities, Vibe).
    The tags will be used while similarity searching.
    
    Output a JSON object strictly:
    {
      "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
    }
    """
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Backstory: {backstory}"}
            ],
            format='json' # Force valid JSON
        )
        data = json.loads(response['message']['content'])
        return data.get('tags', [])
    except Exception as e:
        print(f"⚠️ Error generating tags: {e}")
        return []

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"📂 Loading images from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        images = json.load(f)

    print(f"🏷️  Generating tags for {len(images)} images using {MODEL_NAME}...")

    for img in tqdm(images):
        # Only generate if tags are empty
        if not img.get('tags'):
            backstory = img.get('backstory', "")
            if backstory:
                generated_tags = generate_tags(backstory)
                img['tags'] = generated_tags
            else:
                img['tags'] = []

    print(f"✅ Tagging Complete.")
    print(f"💾 Saving to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
        
    print("\n👉 NEXT STEP: Update 'IMAGE_DATA_FILE' in your 'build_db.py' to point to this new file.")

if __name__ == "__main__":
    main()