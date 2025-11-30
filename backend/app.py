from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama

app = FastAPI()

# --- LOAD RESOURCES ON STARTUP ---
print("Initializing Backend...")
client = chromadb.PersistentClient(path="my_local_db")
chat_col = client.get_collection("chats")
photo_col = client.get_collection("photos")

text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
clip_model = SentenceTransformer('clip-ViT-L-14')

# The Main Chatbot
llm = ChatOllama(model="llama3.2-vision", temperature=0.7)

class Query(BaseModel):
    text: str

@app.post("/chat")
def chat_endpoint(q: Query):
    user_input = q.text.lower()
    
    # --- ROUTER LOGIC ---
    context = ""
    image_result = None

    # 1. Is it a photo request?
    if any(w in user_input for w in ["photo", "pic", "image", "dikhao", "see"]):
        print("Searching Photos...")
        vec = clip_model.encode(q.text)
        results = photo_col.query(query_embeddings=[vec.tolist()], n_results=1)
        if results['ids'][0]:
            filename = results['metadatas'][0][0]['filename']
            image_result = filename
            context = f"[System: User is looking at photo: {filename}]"
    
    # 2. Otherwise search chat memories
    else:
        print("Searching Chats...")
        vec = text_model.encode(q.text)
        results = chat_col.query(query_embeddings=[vec.tolist()], n_results=3)
        if results['documents'][0]:
            memories = "\n".join(results['documents'][0])
            context = f"Relevant Past Memories:\n{memories}"

    # 3. GENERATE REPLY
    prompt = f"""
    System: You are a romantic, funny boyfriend. 
    Context from your relationship: {context}
    User: {q.text}
    """
    
    response = llm.invoke(prompt)
    
    return {
        "reply": response.content,
        "image": image_result
    }

# To run: uvicorn app:app --reload