import streamlit as st
import google.generativeai as genai
import os
import torch
import gc
import json
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline

# ================= CONFIGURATION =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is missing!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
PROMPT_MODEL = genai.GenerativeModel('gemini-2.5-pro')

# File to load poems from
DATA_FILE = "poem.json"

st.set_page_config(page_title="Poem Visualizer", page_icon="🎨", layout="centered")

# ================= 1. HELPER FUNCTIONS =================
def load_writings():
    """Loads the JSON file containing poems/passages."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return []

# ================= 2. MODEL LOADER (SD3.5 Quantized) =================
@st.cache_resource
def load_sd3_model():
    try:
        print("🎨 Loading SD3.5 Medium (NF4 Quantized)...")
        model_id = "stabilityai/stable-diffusion-3.5-medium"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        pipeline.enable_model_cpu_offload()
        return pipeline
    except Exception as e:
        st.error(f"Failed to load SD3.5: {e}")
        return None

# ================= 3. PROMPT ENGINEER =================
def generate_visual_prompt(poem, style):
    # system_prompt = """
    # You are an AI Art Director for Stable Diffusion 3.5. User will provide you with a romantic poem or phrase. You will take some  
    # Task: Convert a romantic poem/passage into a vivid image description for a wallpaper.
    
    # GUIDELINES:
    # 1. Analyze the EMOTION and METAPHORS of the text.
    # 2. Translate metaphors into visual scenes (e.g. "burning love" -> "warm glowing embers, firelight").
    # 3. Use natural language sentences.
    # 4. Focus on Lighting, Composition, and Atmosphere.
    # """
    system_prompt = f"""
    1. Analyzes the poem to find the most 'Picturesque' metaphor/line.
    2. Generates a focused, concise prompt (under 75 words) for SD3.5.
    """
    system_prompt = """
    You are an AI Art Director for Stable Diffusion 3.5.
    
    INPUT: A Poem or Passage.
    TASK: 
    1. Read the text and select the SINGLE most visual, concrete, and picturesque metaphor or scene. 
       (Ignore abstract concepts like "eternal love" or "soul connection").
    2. Write a specific image prompt for ONLY that selected scene.
    
    CONSTRAINTS:
    - **Length:** STRICTLY under 70 words. (To fit the token limit).
    - **Style:** Use natural language sentences.
    - **Focus:** Describe the lighting, the subject, and the atmosphere. 
    
    Example Input: "My love is like a red rose that blooms in winter..."
    Example Output: "A hyper-realistic close-up of a vibrant red rose blooming in the middle of a snowy landscape. Soft morning sunlight reflecting off the snow crystals. Cinematic lighting, macro photography, 8k."
    """
    try:
        response = PROMPT_MODEL.generate_content(f"{system_prompt}\n\nSTYLE: {style}\nTEXT:\n{poem}")
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# ================= 4. UI LAYOUT =================
st.title("🎨 Poem Visualizer")
st.caption("Select a memory from your library and turn it into art.")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh Library"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    st.divider()
    if st.button("🧹 Clear GPU Memory"):
        gc.collect()
        torch.cuda.empty_cache()
        st.toast("GPU Memory Cleared!")

# --- A. DATA LOADING ---
writings = load_writings()
selected_text_content = ""

if writings:
    # Create a list of titles for the dropdown
    # We add a "Custom" option at the top
    options = ["-- Write Custom --"] + [f"({w['type']})" for w in writings]
    
    selection = st.selectbox("📖 Choose a Poem/Passage:", options)
    
    if selection != "-- Write Custom --":
        # Find the selected object based on index or title match
        # (Simple index matching since we added one item to start)
        index = options.index(selection) - 1
        selected_text_content = writings[index]['content']
else:
    st.info(f"💡 Create a '{DATA_FILE}' file to load your saved poems automatically.")

# --- B. INPUT AREA ---
# We pre-fill this text area if a selection was made
poem_text = st.text_area(
    "Edit Text / Prompt:", 
    value=selected_text_content, 
    height=200,
    placeholder="Select a poem above or write here..."
)

# --- C. STYLE SELECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    style = st.selectbox("Choose an Art Style:", [
        "Cinematic & Realistic (Default)",
        "Watercolor Painting",
        "Studio Ghibli Anime",
        "Cyberpunk / Neon",
        "Oil Painting (Impressionist)",
        "Minimalist Vector Art",
        "Dreamy & Ethereal",
        "Dark Fantasy",
        "Vintage Polaroid"
    ])

# --- D. GENERATION ---
if st.button("✨ Visualize this Memory", width="stretch"):
    if not poem_text:
        st.warning("Please provide some text!")
        st.stop()

    # 1. Generate Prompt
    with st.status("🧠 Designing the visual concept...") as status:
        sd_prompt = generate_visual_prompt(poem_text, style)
        status.write(f"**AI Prompt:** {sd_prompt}")
        status.update(label="Concept locked!", state="complete")

    # 2. Generate Image
    with st.spinner("🎨 Painting with SD3.5 (Quantized)..."):
        try:
            pipe = load_sd3_model()
            if pipe:
                image = pipe(
                    prompt=sd_prompt, 
                    num_inference_steps=40, 
                    guidance_scale=4.5,     
                    max_sequence_length=512 
                ).images[0]
                
                st.success("Here is your wallpaper!")
                st.image(image, width="stretch")
                
                # Download Logic
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Wallpaper (HQ)",
                    data=byte_im,
                    file_name="memory_wallpaper.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Generation Error: {e}")