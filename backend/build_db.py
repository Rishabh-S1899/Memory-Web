import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm

# --- CONFIG ---
PHOTO_FOLDER = "photos"
DB_PATH = "my_local_db"

# 1. SETUP MODELS & DB
print("Loading Models (This runs on CPU/GPU automatically)...")
# Text Model (Hinglish/Multilingual)
text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# Image Model (CLIP)
clip_model = SentenceTransformer('clip-ViT-L-14')

client = chromadb.PersistentClient(path=DB_PATH)
chat_col = client.get_or_create_collection("chats")
photo_col = client.get_or_create_collection("photos")

# 2. EMBED MEMORIES
if os.path.exists("memories.json"):
    print("Indexing Memories...")
    with open("memories.json", "r", encoding="utf-8") as f:
        memories = json.load(f)
    
    if memories:
        # Create IDs
        ids = [str(i) for i in range(len(memories))]
        # Compute Embeddings
        embeddings = text_model.encode(memories, show_progress_bar=True)
        # Add to DB
        chat_col.add(documents=memories, embeddings=embeddings, ids=ids)

# 3. EMBED PHOTOS
if os.path.exists(PHOTO_FOLDER):
    print("Indexing Photos...")
    images = [f for f in os.listdir(PHOTO_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(images):
        try:
            path = os.path.join(PHOTO_FOLDER, img_file)
            # CLIP Encode
            emb = clip_model.encode(Image.open(path))
            # Add to DB
            photo_col.add(
                ids=[img_file], 
                embeddings=[emb.tolist()],
                metadatas=[{"filename": img_file}]
            )
        except Exception as e:
            print(f"Error reading {img_file}: {e}")

print("✅ Database built successfully in 'my_local_db' folder!")