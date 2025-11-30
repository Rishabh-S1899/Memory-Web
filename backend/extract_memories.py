import json
from langchain_ollama import ChatOllama
from tqdm import tqdm

# --- CONFIG ---
# Make sure your whatsapp export file is named '_chat.txt'
INPUT_FILE = "_chat.txt" 
OUTPUT_FILE = "memories.json"

# Initialize Local Ollama
llm = ChatOllama(model="mistral-nemo", temperature=0.2)

print("🚀 Starting memory extraction...")

# 1. READ FILE
print(f"Reading {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines read: {len(lines)}")
# 2. CHUNK MESSAGES
chunks = []
current_chunk = []
for line in lines:
    if len(line.strip()) > 10: # Skip short noise
        current_chunk.append(line.strip())
    if len(current_chunk) >= 25:
        chunks.append("\n".join(current_chunk))
        current_chunk = []

# 3. SYSTEM PROMPT (Hinglish Optimized)
system_prompt = """
You are an expert Relationship Analyst. You understand Hinglish (Hindi-English).
Task: Extract romantic memories, milestones, future promises, or funny inside jokes from this chat chunk.
Ignore: Logistics (traffic, food orders), "Good morning", or boring chitchat.
Output: A pure JSON list of strings. Example: ["We went to Marine Drive", "She called me cute"]
"""

# 4. PROCESS
print("Extracting memories (This depends on your GPU speed)...")
all_memories = []

for chunk in tqdm(chunks):
    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", chunk)
        ])
        content = response.content
        
        # Extract JSON
        if "[" in content and "]" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            json_str = content[start:end]
            all_memories.extend(json.loads(json_str))
    except Exception as e:
        print(f"Skipped chunk: {e}")

# 5. SAVE
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_memories, f, indent=2)

print(f"✅ Success! Saved {len(all_memories)} memories to {OUTPUT_FILE}")