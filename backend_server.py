import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import logging
from datetime import datetime
import torch
import google.generativeai as genai
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from io import BytesIO

# ================= 1. CONFIGURATION =================
# CRITICAL: Point this to where your actual image files are stored on disk
IMAGE_DIRECTORY_ROOT = "./public/hero-images" 
DB_PATH = "./backendv2/chroma_db_data"
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v4"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing!")

genai.configure(api_key=GEMINI_API_KEY)


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


# ================= 2. API SETUP =================
app = FastAPI()

# Allow React to talk to this Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Images: http://localhost:8000/static/filename.jpg
if not os.path.exists(IMAGE_DIRECTORY_ROOT):
    os.makedirs(IMAGE_DIRECTORY_ROOT, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMAGE_DIRECTORY_ROOT), name="static")

# ================= 3. MODELS & AGENTS =================
print("🧠 Loading Agents...")
# Using the EXACT models from your script
ROUTER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
EXPANDER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
FILTER_MODEL = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
SOLVER_MODEL = genai.GenerativeModel('gemini-2.5-pro')
PROMPT_MODEL = genai.GenerativeModel('gemini-2.5-pro') # For Wallpapers

# Database Setup
class JinaEmbedder(EmbeddingFunction):
    def __init__(self, model_name, device="cuda"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, task="retrieval", prompt_name="query", show_progress_bar=False).tolist()

def load_db():
    logger.info("Initializing Database connection...")
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = JinaEmbedder(model_name=EMBEDDING_MODEL_NAME, device="cuda")
    chat_col = client.get_collection(name="chat_memories", embedding_function=emb_fn)
    img_col = client.get_collection(name="image_memories", embedding_function=emb_fn)
    logger.info("Database loaded successfully.")
    return chat_col, img_col


try:
    print("📂 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = JinaEmbedder(EMBEDDING_MODEL_NAME, device="cuda")
    chat_col = client.get_collection(name="chat_memories", embedding_function=emb_fn)
    img_col = client.get_collection(name="image_memories", embedding_function=emb_fn)
    print("✅ Database Connected")
except Exception as e:
    print(f"❌ DB Error: {e}")

# Stable Diffusion (Lazy Loading)
sd3_pipeline = None

def get_sd3_pipeline():
    global sd3_pipeline
    if sd3_pipeline is None:
        print("🎨 Loading SD3.5 (NF4)...")
        model_id = "stabilityai/stable-diffusion-3.5-medium"
        nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model_nf4 = SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer", quantization_config=nf4_config, torch_dtype=torch.bfloat16)
        sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, transformer=model_nf4, torch_dtype=torch.bfloat16)
        sd3_pipeline.enable_model_cpu_offload()
    return sd3_pipeline

# ================= 4. HELPER FUNCTIONS =================
# def convert_path_to_url(abs_path):
#     """Converts local file path to localhost URL for the frontend"""
#     filename = os.path.basename(abs_path)
#     return f"http://localhost:8000/static/{filename}"
# ================= 4. HELPER FUNCTIONS =================
def convert_path_to_url(db_path):
    """
    1. Extracts filename from the old DB path (e.g. 'C:\\old\\img.jpg' -> 'img.jpg')
    2. Checks if 'img.jpg' exists in the new IMAGE_DIRECTORY_ROOT
    3. Returns the correct localhost URL
    """
    if not db_path: return None
    
    # Extract just the filename (Works on Windows & Linux paths)
    filename = os.path.basename(db_path)
    
    # Construct the NEW physical path
    real_path = os.path.join(IMAGE_DIRECTORY_ROOT, filename)
    print('This is real_path: ',real_path)
    # Verify existence (Optional but good for safety)
    if os.path.exists(real_path):
        # Return the URL that React can load
        return f"http://localhost:8000/static/{filename}"
    else:
        print(f"⚠️ Warning: Image not found at {real_path}")
        return None



# ================= 5. AGENT LOGIC (EXACTLY AS PROVIDED) =================

def agent_router(query):
    logger.info(f"--- AGENT 1: ROUTER --- Input: '{query}'")
    system_prompt = """
    You are a Search Weight Manager.
    User Query: {query}
    
    Task: Assign relevance weights (0.0 to 1.0) to Text vs Images.
    
    GUIDELINES:
    1. "Show me photos", "How did we look?" -> image_weight: 0.9, text_weight: 0.1
    2. "What did I say?", "Quote the message" -> image_weight: 0.1, text_weight: 1.0
    3. "Tell me about the trip", "How was March?" -> image_weight: 0.4, text_weight: 0.6 (Narrative needs both)
    
    Output JSON: {"text_weight": float, "image_weight": float, "reasoning": "string"}
    """
    try:
        response = ROUTER_MODEL.generate_content(f"{system_prompt}\nQUERY: {query}")
        decision = json.loads(response.text)
        logger.info(f"Router Decision: {decision}")
        return decision
    except Exception as e:
        logger.error(f"Router Failed: {e}")
        # return {"search_chats": True, "search_images": True, "reasoning": "Fallback Error"}
        return {"text_weight": 0.8, "image_weight": 0.4, "reasoning": "Error fallback"}

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
        # if decision['search_chats']:
            chat_res = chat_col.query(query_texts=[q], n_results=20)
            if chat_res['documents']:
                for i, doc in enumerate(chat_res['documents'][0]):
                    meta = chat_res['metadatas'][0][i]
                    uid = meta.get('chunk_id', doc[:10])
                    dist = chat_res['distances'][0][i]
                    
                    if uid not in seen_chat_ids and dist < 1.5:
                        seen_chat_ids.add(uid)
                        date = meta.get('date', 'Unknown')
                        context_text += f"--- MEMORY ({date}) ---\n{doc}\nRAW: {meta.get('raw_chat_dump', '')}\n\n"
                        debug_retrieval_log.append(f"Chat: {date} (Dist: {dist:.2f})")

        # 2. Search Images
        # if decision['search_images']:
            img_res = img_col.query(query_texts=[q], n_results=2)
            if img_res['documents']:
                for i, doc in enumerate(img_res['documents'][0]):
                    meta = img_res['metadatas'][0][i]
                    path = meta['filepath']
                    # print('This is the path: ',path)
                    # new_path=path.split(r'\\')
                    # print('This is the new path before joining: ',new_path)
                    # new_path= os.path.join(IMAGE_DIRECTORY_ROOT, new_path[-1])
                    # path=new_path
                    # print('This is the new path: ',new_path)
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

def agent_solver(query, context,weights):
    logger.info("--- AGENT 5: SOLVER --- Synthesizing...")

    # DYNAMIC INSTRUCTION BASED ON WEIGHTS
    w_text = weights.get('text_weight', 0.7)
    w_img = weights.get('image_weight', 0.3)
    
    focus_instruction = ""
    if w_img > 0.8 and w_text < 0.5:
        focus_instruction = """
        **MODE: VISUAL GALLERY**
        - The user primarily wants to see photos.
        - Your text response should be brief.
        - Describe the photos vividy.
        """
    elif w_text > 0.8 and w_img < 0.5:
        focus_instruction = """
        **MODE: DEEP CONVERSATION**
        - The user wants details about what was SAID.
        - You MUST quote the "TRANSCRIPT" sections.
        - Use photos sparingly, only if they perfectly match a specific message.
        - Focus on the text side, include photos only if they are a direct match with the query.
        """
    else:
        focus_instruction = """
        **MODE: NARRATIVE STORY**
        - Create a timeline.
        - Weave the Chat Logs (Facts/Feelings) and Photos (Visuals) together equally.
        """

    system_prompt = f"""
    You are an ancient Storyteller for Rishabh and Chetna. You have access to their history, both text and photos.
    Answer as a wizard from the past, weaving memories into a heartfelt narrative.
    INPUT DATA:
    Context dump (from chats and image descriptions):
    TASK: Answer the user's question.
    
    {focus_instruction}

    RULES:
    1. Filter: Use dates to ignore irrelevant logs in the context dump.
    2. Photos: Mention "[I found a photo of this!]" if relevant.
    3. Quotes: Use RAW CHAT logs for nostalgia.
    4. Tone: Warm, loving, Hinglish.
    """
    full_prompt = f"{system_prompt}\n\nQUERY: {query}\n\nCONTEXT:\n{context}"
    response= SOLVER_MODEL.generate_content(full_prompt, stream=False)
    return response.text

# ================= 6. API ENDPOINTS =================

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    text: str
    images: list[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"📩 Query: {req.message}")
    
    # 1. Router
    decision = agent_router(req.message)
    
    # 2. Expander
    queries = agent_expander(req.message, decision)
    
    # 3. Retrieval
    context_txt, raw_imgs, log = run_retrieval(queries, decision)
    
    # 4. Filter
    filtered_paths = agent_image_filter(req.message, raw_imgs)
    
    # 5. Solver
    story = agent_solver(req.message, context_txt, decision)
    
    # Convert local paths to URLs
    print('filtered paths: ',filtered_paths)
    image_urls = [convert_path_to_url(p) for p in filtered_paths]
    print('These are image_urls: ',image_urls)
    
    return ChatResponse(text=story, images=image_urls)

# --- WALLPAPER ENDPOINT ---
class WallpaperRequest(BaseModel):
    text: str
    style: str

class WallpaperResponse(BaseModel):
    image_url: str

@app.post("/generate-wallpaper", response_model=WallpaperResponse)
async def generate_wallpaper(req: WallpaperRequest):
    print(f"🎨 Visualizing: {req.style}")
    
    # 1. Generate Prompt (SD3.5 style)
    prompt_sys = """
    You are an AI Art Director for Stable Diffusion 3.5.
    Task: Convert a romantic poem/passage into a detailed image prompt.
    RULES: Use natural language. Focus on lighting and atmosphere. Keep under 70 words.
    """
    prompt_resp = PROMPT_MODEL.generate_content(f"{prompt_sys}\nSTYLE: {req.style}\nTEXT: {req.text}")
    sd_prompt = prompt_resp.text.strip()
    
    # 2. Generate Image
    try:
        pipe = get_sd3_pipeline()
        image = pipe(
            prompt=sd_prompt, 
            num_inference_steps=40, 
            guidance_scale=4.5, 
            max_sequence_length=512
        ).images[0]
        
        # 3. Save
        filename = f"gen_{datetime.now().strftime('%H%M%S')}.png"
        save_path = os.path.join(IMAGE_DIRECTORY_ROOT, "generated", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        
        # Note: Since we mounted IMAGE_DIRECTORY_ROOT to /static, if we save to /data/generated
        # the URL is /static/generated/filename.png
        image_url = f"http://localhost:8000/static/generated/{filename}"
        
        return WallpaperResponse(image_url=image_url)

    except Exception as e:
        print(f"❌ Gen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)