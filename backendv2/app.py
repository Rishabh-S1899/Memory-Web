import streamlit as st
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import os

# ================= CONFIGURATION =================
DB_PATH = "./chroma_db_data"
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v4"

# 🔑 API KEY
# Replace with your actual key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# MODEL CONFIGURATION
# Flash for speed (Router/Expander), Pro for quality (Final Answer)
ROUTER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
EXPANDER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
SOLVER_MODEL = genai.GenerativeModel('gemini-2.5-pro') # The "Brain"

# =================================================

st.set_page_config(page_title="Anniversary Bot", page_icon="❤️", layout="centered")

# --- CUSTOM JINA WRAPPER ---
class JinaEmbedder(EmbeddingFunction):
    def __init__(self, model_name, device="cpu"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    def __call__(self, input: Documents) -> Embeddings:
        # Prompt name is 'query' for retrieval
        return self.model.encode(input, task="retrieval", prompt_name="query", show_progress_bar=False).tolist()

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = JinaEmbedder(model_name=EMBEDDING_MODEL_NAME, device="cpu")
    chat_col = client.get_collection(name="chat_memories", embedding_function=emb_fn)
    img_col = client.get_collection(name="image_memories", embedding_function=emb_fn)
    return chat_col, img_col

try:
    chat_col, img_col = load_db()
except Exception as e:
    st.error(f"DB Error: {e}")
    st.stop()

# ================= AGENT 1: THE ROUTER =================
def agent_router(query):
    """
    Decides WHERE to look.
    Output: JSON {'search_chats': bool, 'search_images': bool, 'reasoning': str}
    """
    system_prompt = """
    You are a Search Router for a relationship database.
    Analyze the User Query.
    
    RULES:
    - If user asks about visual things (looks, clothes, places, photos), set "search_images": true.
    - If user asks about conversations, facts, dates, feelings, inside jokes, set "search_chats": true.
    - If vague (e.g., "Tell me about us"), search BOTH.
    
    Output JSON: {"search_chats": bool, "search_images": bool}
    """
    
    response = ROUTER_MODEL.generate_content(f"{system_prompt}\nQUERY: {query}")
    try:
        return json.loads(response.text)
    except:
        return {"search_chats": True, "search_images": True} # Fallback

# ================= AGENT 2: THE EXPANDER =================
def agent_expander(query, route_decision):
    """
    Expands the query into better keywords for Vector Search.
    Output: JSON {'expanded_queries': [list of strings]}
    """
    # Only expand if we are searching chats (images usually need literal queries)
    if not route_decision['search_chats']:
        return [query]

    system_prompt = """
    You are a Query Optimizer for a vector database containing WhatsApp chats.
    The user might ask vague questions. Generate 3-5 specific search queries to find the answer.
    
    STRATEGY:
    1. If asking for "Jokes/Funny", add: "Hahaha", "Lmao", "Roast", "Teasing".
    2. If asking for "Fights", add: "Sorry", "Gussa", "Angry", "Upset".
    3. If asking for "Love", add: "Love you", "Miss you", "Heart".
    4. Keep the original query as the first item.
    
    User Query: {query}
    Output JSON: {"expanded_queries": ["query1", "query2", ...]}
    """
    
    response = EXPANDER_MODEL.generate_content(f"{system_prompt}\nQUERY: {query}")
    try:
        return json.loads(response.text)['expanded_queries']
    except:
        return [query]

# ================= AGENT 3: THE RETRIEVER =================
def run_retrieval(expanded_queries, decision):
    context_text = ""
    images_found = []
    seen_ids = set()

    # A. Search Chats (Using Expanded Queries)
    if decision['search_chats']:
        # We query Chroma with ALL expanded variations at once
        results = chat_col.query(query_texts=expanded_queries, n_results=5)
        
        # Flatten results (List of lists)
        if results['documents']:
            for i, sublist in enumerate(results['documents']):
                for j, doc in enumerate(sublist):
                    meta = results['metadatas'][i][j]
                    dist = results['distances'][i][j]
                    uid = meta.get('chunk_id', doc[:10])
                    
                    if uid not in seen_ids and dist < 0.85: # Threshold
                        seen_ids.add(uid)
                        date = meta.get('date', 'Unknown')
                        raw = meta.get('raw_chat_dump', 'No raw log')
                        context_text += f"[Memory {date}]: {doc}\nRAW LOG: {raw}\n---\n"

    # B. Search Images (Using Original Query only - visuals are literal)
    if decision['search_images']:
        # We use the first query (original) for images
        img_results = img_col.query(query_texts=[expanded_queries[0]], n_results=3)
        
        if img_results['documents']:
            for i, doc in enumerate(img_results['documents'][0]):
                meta = img_results['metadatas'][0][i]
                dist = img_results['distances'][0][i]
                
                if dist < 0.65: # Stricter for images
                    images_found.append(meta['filepath'])
                    context_text += f"[Photo Info ({meta['date']})]: {doc}\n"

    return context_text, images_found

# ================= AGENT 4: THE SYNTHESIZER (FINAL) =================
def agent_solver(query, context, images):
    system_prompt = """
    You are a romantic and witty AI assistant for Rishabh and Chetna's anniversary.
    You have access to retrieved memories.
    
    STRICT RULES:
    1. GROUNDING: Answer ONLY using the Context provided. If the context is empty, say "I couldn't find a memory about that specifically, but I know you guys are awesome."
    2. TONE: Warm, nostalgic, slightly playful. Use Hinglish if the logs are Hinglish.
    3. QUOTING: If you see 'RAW LOG', quote the funny/cute parts directly. 
    4. IMAGES: If 'Photo Info' is present, explicitly mention: "I found a photo from that day!"
    
    """
    
    full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}\n\nCONTEXT:\n{context}"
    
    # Using the stream to reduce perceived latency
    return SOLVER_MODEL.generate_content(full_prompt, stream=True)

# ================= UI LOGIC =================
st.title("❤️ Time Capsule (Multi-Agent)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready. Ask me anything!"}]

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "images" in msg and msg["images"]:
            cols = st.columns(len(msg["images"]))
            for i, path in enumerate(msg["images"]):
                if os.path.exists(path):
                    cols[i].image(path, width=200)

# Input Loop
if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        
        # 1. ROUTING
        status.caption("🧠 Agent 1: Routing...")
        decision = agent_router(prompt)
        
        # 2. EXPANSION
        status.caption("🔍 Agent 2: Optimizing Search...")
        expanded_queries = agent_expander(prompt, decision)
        # Optional: Show what the bot is searching for
        # st.toast(f"Searching for: {expanded_queries}") 
        
        # 3. RETRIEVAL
        status.caption("📚 Agent 3: Reading Database...")
        context_txt, imgs = run_retrieval(expanded_queries, decision)
        
        # 4. SYNTHESIS
        status.caption("✨ Agent 4: Writing...")
        response_stream = agent_solver(prompt, context_txt, imgs)
        
        # Output
        status.empty() # Clear status text
        
        def stream_parser(stream):
            for chunk in stream:
                if chunk.text: yield chunk.text
                
        full_response = st.write_stream(stream_parser(response_stream))
        
        # Display Images
        if imgs:
            cols = st.columns(len(imgs))
            for i, path in enumerate(imgs):
                if os.path.exists(path):
                    cols[i].image(path, caption="Found this memory", use_container_width=True)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response, "images": imgs})