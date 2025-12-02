import streamlit as st
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import os
import logging
import time
from datetime import datetime

# ================= 1. LOGGING SETUP =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= 2. CONFIGURATION =================
DB_PATH = "./chroma_db_data"
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v4"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is missing! Set it in your environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# MODELS
ROUTER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
EXPANDER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
FILTER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"}) # <--- FILTER AGENT
SOLVER_MODEL = genai.GenerativeModel('gemini-2.5-pro')

st.set_page_config(page_title="Anniversary Bot", page_icon="❤️", layout="wide")

# ================= 3. DB SETUP =================
class JinaEmbedder(EmbeddingFunction):
    def __init__(self, model_name, device="cuda"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, task="retrieval", prompt_name="query", show_progress_bar=False).tolist()

@st.cache_resource
def load_db():
    logger.info("Initializing Database connection...")
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = JinaEmbedder(model_name=EMBEDDING_MODEL_NAME, device="cuda")
    chat_col = client.get_collection(name="chat_memories", embedding_function=emb_fn)
    img_col = client.get_collection(name="image_memories", embedding_function=emb_fn)
    logger.info("Database loaded successfully.")
    return chat_col, img_col

try:
    chat_col, img_col = load_db()
except Exception as e:
    st.error(f"DB Error: {e}")
    st.stop()

# ================= 4. AGENTS =================

def agent_router(query):
    logger.info(f"--- AGENT 1: ROUTER --- Input: '{query}'")
    system_prompt = """
    You are a Search Scope Manager.
    User Query: {query}
    DECISION LOGIC:
    1. STRICTLY VISUAL: "Show me photos" -> search_images: True, search_chats: False
    2. STRICTLY TEXTUAL: "Quote the message" -> search_images: False, search_chats: True
    3. NARRATIVE/MEMORY (Default): "How was March?", "Tell me about the trip" -> search_images: True, search_chats: True
    
    Output JSON: {"search_chats": bool, "search_images": bool, "reasoning": "string"}
    """
    try:
        response = ROUTER_MODEL.generate_content(f"{system_prompt}\nQUERY: {query}")
        decision = json.loads(response.text)
        logger.info(f"Router Decision: {decision}")
        return decision
    except Exception as e:
        logger.error(f"Router Failed: {e}")
        return {"search_chats": True, "search_images": True, "reasoning": "Fallback Error"}

def agent_expander(query, route_decision):
    logger.info(f"--- AGENT 2: EXPANDER --- Input: '{query}'")
    current_year = datetime.now().year
    
    system_prompt = f"""
    You are a Query Generator.
    User Query: "{query}"
    Context Year: {current_year}
    
    Task: Generate 3-5 distinct search strings to maximize recall. 
    Create the queries such that it is easier to retrieve relevant memories.
    Do not necessarily shorten the query; focus on variety.
    For example, some of the tags from mny data are : [
      "Humayun's Tomb",
      "first date",
      "awkward",
      "Delhi",
      "travel",
      "historical site",
      "couple",
      "romantic"
    ]
    This is for the image with description "Awkward first date photos at Humayun's Tomb"
    The above are just examples, do not output them directly.


    Understand and then make the relevant queries
    STRATEGIES:
    - If date mentioned ("March"): Generate "March {current_year}", "March {current_year-1}", "Events in March".
    - If topic mentioned ("Fights"): Generate "Arguments", "Sorry", "Upset", "Misunderstanding".
    - If "Inside Jokes": Generate "Hahaha", "Lmao", "Teasing", "Funny".
    
    Output JSON: {{ "queries": ["q1", "q2", "q3"] }}
    """
    try:
        response = EXPANDER_MODEL.generate_content(system_prompt)
        queries = json.loads(response.text)['queries']
        logger.info(f"Expanded Queries: {queries}")
        return queries
    except Exception as e:
        logger.error(f"Expander Failed: {e}")
        return [query]

def run_retrieval(queries, decision):
    logger.info(f"--- AGENT 3: RETRIEVER --- Running {len(queries)} sub-queries...")
    context_text = ""
    
    # MODIFIED: Store full dicts instead of just paths, so Filter Agent can read descriptions
    img_candidates = [] 
    
    seen_chat_ids = set()
    seen_img_ids = set()
    debug_retrieval_log = []

    for q in queries:
        # 1. Search Chats
        if decision['search_chats']:
            chat_res = chat_col.query(query_texts=[q], n_results=5)
            if chat_res['documents']:
                for i, doc in enumerate(chat_res['documents'][0]):
                    meta = chat_res['metadatas'][0][i]
                    uid = meta.get('chunk_id', doc[:10])
                    dist = chat_res['distances'][0][i]
                    
                    if uid not in seen_chat_ids and dist < 0.6:
                        seen_chat_ids.add(uid)
                        date = meta.get('date', 'Unknown')
                        context_text += f"--- MEMORY ({date}) ---\n{doc}\nRAW: {meta.get('raw_chat_dump', '')}\n\n"
                        debug_retrieval_log.append(f"Chat: {date} (Dist: {dist:.2f})")

        # 2. Search Images
        if decision['search_images']:
            img_res = img_col.query(query_texts=[q], n_results=2)
            if img_res['documents']:
                for i, doc in enumerate(img_res['documents'][0]):
                    meta = img_res['metadatas'][0][i]
                    path = meta['filepath']
                    dist = img_res['distances'][0][i]
                    
                    if path not in seen_img_ids and dist < 1.0: # High Recall Threshold
                        seen_img_ids.add(path)
                        
                        # Store Metadata for Filter Agent
                        img_obj = {
                            "path": path,
                            "date": meta.get('date', 'Unknown'),
                            "desc": doc
                        }
                        img_candidates.append(img_obj)
                        
                        context_text += f"--- PHOTO AVAILABLE ({img_obj['date']}) ---\nDesc: {doc}\nFile: {path}\n\n"
                        debug_retrieval_log.append(f"Img: {path} (Dist: {dist:.2f})")

    logger.info(f"Retrieval Found: {len(seen_chat_ids)} chats, {len(img_candidates)} images")
    return context_text, img_candidates, debug_retrieval_log

# ================= NEW: FILTER AGENT =================
def agent_image_filter(query, img_candidates):
    logger.info("--- AGENT 4: FILTER --- Verification starting...")
    if not img_candidates: return []

    # Format list for LLM
    candidates_text = ""
    for i, img in enumerate(img_candidates):
        candidates_text += f"Image {i}: [Date: {img['date']}] Description: {img['desc']}\n"

    system_prompt = f"""
    You are a Content Filter.
    User Query: "{query}"
    
    Task: Return indices of images that are RELEVANT to the query.
    Look at the date and the backstory for each image and then decide if it fits the user's request.
    RULES:
    1. If query specifies a Date (e.g. "March"), remove images from wrong months.
    2. If query specifies an Activity (e.g. "Eating"), remove images of "Sleeping".
    3. If the query is broad, then think about overall relevance.
    
    Output JSON: {{ "valid_indices": [0, 2], "reasoning": "string" }}
    """
    
    try:
        response = FILTER_MODEL.generate_content(f"{system_prompt}\nCANDIDATES:\n{candidates_text}")
        indices = json.loads(response.text)['valid_indices']
        
        # Map indices back to paths
        valid_paths = [img_candidates[i]['path'] for i in indices if i < len(img_candidates)]
        logger.info(f"Filter passed {len(valid_paths)}/{len(img_candidates)} images.")
        return valid_paths
    except Exception as e:
        logger.error(f"Filter Agent Failed: {e}")
        return [x['path'] for x in img_candidates] # Fallback: return all

def agent_solver(query, context):
    logger.info("--- AGENT 5: SOLVER --- Synthesizing...")
    system_prompt = """
    You are an ancient Storyteller for Rishabh and Chetna. You have access to their history, both text and photos.
    Answer as a wizard from the past, weaving memories into a heartfelt narrative.
    INPUT: Memories (Text/Photos).
    TASK: Answer the user's question.
    
    RULES:
    1. Filter: Use dates to ignore irrelevant logs in the context dump.
    2. Photos: Mention "[I found a photo of this!]" if relevant.
    3. Quotes: Use RAW CHAT logs for nostalgia.
    4. Tone: Warm, loving, Hinglish.
    """
    full_prompt = f"{system_prompt}\n\nQUERY: {query}\n\nCONTEXT:\n{context}"
    return SOLVER_MODEL.generate_content(full_prompt, stream=True)

# ================= 5. UI LAYOUT =================
st.title("❤️ Anniversary Time Capsule")

# Helper to simulate streaming text
def simulate_stream(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) 

col_chat, col_debug = st.columns([2, 1])

# --- DEBUG SIDEBAR ---
with col_debug:
    st.header("🔧 Agent Pipeline")
    expander_router = st.expander("🧠 1. Router", expanded=False)
    expander_expander = st.expander("🔍 2. Expander", expanded=False)
    expander_retrieval = st.expander("📚 3. Retrieval Logs", expanded=False)
    expander_filter = st.expander("🖼️ 4. Image Filter", expanded=False) # <--- Added Filter Log
    expander_solver = st.expander("✨ 5. Solver Context", expanded=False)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready! Ask me anything."}]

# --- CHAT AREA ---
with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "images" in msg and msg["images"]:
                # --- GRID LAYOUT FOR HISTORY ---
                cols = st.columns(3)
                for i, path in enumerate(msg["images"]):
                    if os.path.exists(path):
                        cols[i % 3].image(path, use_container_width=True)

    if prompt := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            # 1. Router
            decision = agent_router(prompt)
            expander_router.json(decision)
            
            # 2. Expander
            queries = agent_expander(prompt, decision)
            expander_expander.write(queries)
            
            # 3. Retrieval (Now returns full img objects)
            context_txt, raw_img_data, debug_logs = run_retrieval(queries, decision)
            expander_retrieval.code("\n".join(debug_logs), language="text")
            
            # 4. Filter (The New Step)
            with st.spinner("Verifying Images..."):
                filtered_img_paths = agent_image_filter(prompt, raw_img_data)
                
            # Log Filter decisions
            expander_filter.write(f"**Input:** {len(raw_img_data)} images")
            expander_filter.write(f"**Output:** {len(filtered_img_paths)} images")
            if len(raw_img_data) > len(filtered_img_paths):
                expander_filter.caption("Some images were removed as irrelevant.")
            
            # 5. Solving
            response_stream = agent_solver(prompt, context_txt)
            
            full_response = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            # 6. Display Images (Using Filtered Results)
            if filtered_img_paths:
                # st.write("---")
                # # Fix: Always 3 columns
                # cols = st.columns(3) 
                # for i, path in enumerate(filtered_img_paths):
                #     if os.path.exists(path):
                #         # Fix: use_container_width instead of width=True
                #         cols[i % 3].image(path, caption="Memory found", width=True)
                st.write("---")
                st.caption(f"Found {len(filtered_img_paths)} photos:")

                # Create a fixed grid of 3 columns
                cols = st.columns(3)

                for i, path in enumerate(filtered_img_paths):
                    if os.path.exists(path):
                        # Calculate which column index (0, 1, or 2)
                        col_index = i % 3

                        with cols[col_index]:
                            try:
                                st.image(
                                    path, 
                                    caption=f"Memory {i+1}", 
                                    use_container_width=True # Let Streamlit handle resizing safely
                                )
                            except Exception as e:
                                st.error(f"Img Error: {e}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "images": filtered_img_paths
            })