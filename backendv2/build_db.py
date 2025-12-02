import json
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import torch

# ================= CONFIGURATION =================
CHAT_DATA_FILE = "gemma_knowledge_base_enriched_final.json"
IMAGE_DATA_FILE = "images_tagged.json"
DB_PATH = "./chroma_db_data" 

# Jina V4 Model Name
MODEL_NAME = "jinaai/jina-embeddings-v4"
# =================================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def date_to_int(date_str):
    # Converts "2024-03-01" -> 20240301 (Integer)
    try:
        clean_str = date_str.replace("-", "").replace("/", "").split(" ")[0]
        return int(clean_str)
    except:
        return 0 # Fallback for unknown dates



# --- CUSTOM JINA WRAPPER FOR CHROMA ---
class JinaEmbedder(EmbeddingFunction):
    def __init__(self, model_name, device="cuda"):
        print(f"🧠 Loading Jina Model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        
    def __call__(self, input: Documents) -> Embeddings:
        # We enforce specific parameters for Jina V4
        # Since we are BUILDING the DB, these are "passages", not queries.
        embeddings = self.model.encode(
            sentences=input,
            task="retrieval",
            prompt_name="passage", # <--- CRITICAL: Tells model this is data to be stored
            show_progress_bar=False
        )
        # Convert numpy array to list for Chroma
        return embeddings.tolist()

def normalize_date(date_str):
    try:
        if len(date_str) == 10: return date_str
        return date_str.split(" ")[0]
    except: return "Unknown"

def main():
    # 1. Setup Device & Client
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing ChromaDB on {device.upper()}...")
    
    # Resetting the DB folder is recommended when switching embedding models!
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        print("⚠️ NOTE: Make sure you have deleted the old 'chroma_db_data' folder before running this!")
    
    client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Initialize Custom Jina Embedder
    # This instance is configured for INDEXING (prompt_name="passage")
    emb_fn = JinaEmbedder(model_name=MODEL_NAME, device=device)

    # ================= PROCESS CHATS =================
    print(f"📂 Loading Chat Data from {CHAT_DATA_FILE}...")
    try:
        with open(CHAT_DATA_FILE, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        chat_collection = client.get_or_create_collection(
            name="chat_memories",
            embedding_function=emb_fn, # Pass our custom class
            metadata={"description": "WhatsApp history"}
        )

        ids, documents, metadatas = [], [], []
        seen_ids = set()

        print("⚡ Vectorizing Chat Memories with Jina V4...")
        for item in tqdm(chat_data):
            # Duplicate ID Fix
            base_id = item['metadata']['chunk_id']
            unique_id = base_id
            counter = 1
            while unique_id in seen_ids:
                unique_id = f"{base_id}_dup{counter}"
                counter += 1
            seen_ids.add(unique_id)

            # Keyword Handling
            keywords = item['metadata'].get('keywords', [])
            if keywords is None: keywords = []
            
            # Construct Text
            summary = item.get('summary_vector_source', '')
            vector_text = f"{summary} Keywords: {', '.join(keywords)}"
            
            # meta = {
            #     "date": normalize_date(item['metadata'].get('date', "2024-01-01")),
            #     "type": "chat",
            #     "score": item['metadata'].get('score', 5),
            #     "mood": item['metadata'].get('mood', "Neutral"),
            #     "raw_chat_dump": json.dumps(item['original_chat'], ensure_ascii=False)
            # }
            # ... inside chat loop ...
            date_str = normalize_date(item['metadata'].get('date', "2024-01-01"))

            meta = {
                "date": date_str,         # Keep string for display
                "date_int": date_to_int(date_str), # <--- NEW: Integer for Filtering
                "type": "chat",
                "score": item['metadata'].get('score', 5),
                "mood": item['metadata'].get('mood', "Neutral"),
                "raw_chat_dump": json.dumps(item['original_chat'], ensure_ascii=False)
            }

            ids.append(unique_id)
            documents.append(vector_text)
            metadatas.append(meta)

        if ids:
            chat_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Indexed {len(ids)} chat chunks.")

    except FileNotFoundError:
        print(f"❌ Error: {CHAT_DATA_FILE} not found.")

    # ================= PROCESS IMAGES =================
    print(f"\n🖼️ Loading Image Data from {IMAGE_DATA_FILE}...")
    try:
        with open(IMAGE_DATA_FILE, 'r', encoding='utf-8') as f:
            image_data = json.load(f)

        image_collection = client.get_or_create_collection(
            name="image_memories",
            embedding_function=emb_fn,
            metadata={"description": "Photos"}
        )

        img_ids, img_docs, img_metas = [], [], []

        print("⚡ Vectorizing Images with Jina V4...")
        for item in tqdm(image_data):
            # vector_text = item['backstory']
            # We append the tags to the text so the AI 'reads' them
            tags_list = item.get('tags', [])
            if tags_list is None: tags_list = []

            vector_text = f"{item['backstory']} Context: {', '.join(tags_list)}"            
            tags = item.get('tags', [])
            if tags is None: tags = []

            # meta = {
            #     "date": normalize_date(item['exif_date']),
            #     "type": "image",
            #     "filepath": item['filepath'],
            #     "tags": ",".join(tags)
            # }

            # ... inside image loop ...
            date_str = normalize_date(item['exif_date'])

            meta = {
                "date": date_str,
                "date_int": date_to_int(date_str), # <--- NEW
                "type": "image",
                "filepath": item['filepath'],
                "tags": ",".join(tags)
            }

            img_ids.append(item['id'])
            img_docs.append(vector_text)
            img_metas.append(meta)

        if img_ids:
            image_collection.upsert(ids=img_ids, documents=img_docs, metadatas=img_metas)
            print(f"✅ Indexed {len(img_ids)} images.")
            
    except FileNotFoundError:
        print(f"⚠️ Warning: {IMAGE_DATA_FILE} not found.")

    print("\n✨ Database Build Complete!")

if __name__ == "__main__":
    main()