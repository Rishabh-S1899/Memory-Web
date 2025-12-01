import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import torch
import os
import numpy as np

# ================= CONFIGURATION =================
DB_PATH = "./chroma_db_data"
# =================================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    print(f"🕵️‍♂️ Inspecting Database at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Check CHATS
    try:
        chat_col = client.get_collection("chat_memories")
        count = chat_col.count()
        print(f"\n✅ Chat Collection found! Total Memories: {count}")
        
        data = chat_col.peek(limit=1)
        
        # FIX: explicitly check length to avoid numpy ambiguity
        if len(data['embeddings']) > 0:
            # Check dimension of the first vector
            vec_len = len(data['embeddings'][0])
            print(f"   📐 Vector Dimension: {vec_len} (Jina V4 is usually 768)")
            print(f"   📝 Sample Text: {data['documents'][0][:100]}...")
        else:
            print("   ❌ WARNING: Embeddings list is empty!")
            
    except Exception as e:
        print(f"   ❌ Error reading Chats: {e}")

    # Check IMAGES
    try:
        img_col = client.get_collection("image_memories")
        count = img_col.count()
        print(f"\n✅ Image Collection found! Total Images: {count}")
        
        data = img_col.peek(limit=1)
        if len(data['embeddings']) > 0:
            print(f"   📐 Vector Dimension: {len(data['embeddings'][0])}")
            print(f"   📸 Sample Backstory: {data['documents'][0][:100]}...")
        else:
             print("   ❌ WARNING: Embeddings list is empty!")

    except Exception as e:
        print(f"   ❌ Error reading Images: {e}")

if __name__ == "__main__":
    main()