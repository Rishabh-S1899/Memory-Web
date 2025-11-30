#!/usr/bin/env python3
# build_shared_text_db.py
"""
Build a shared Chroma DB for:
 - chat memories (memories_gold.json)
 - photo backstories (photos_backstories_template.json)

Uses one multilingual SentenceTransformer for both (Option 1).
Saves / loads embedding caches and writes two Chroma collections:
 - 'memories'
 - 'photos_backstory'

Interactive tester at the end to query and inspect top-K results.

Config at the top of the file.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
# MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# If you want higher quality and can afford memory, swap to:
# MODEL_NAME = "intfloat/e5-large-v2"
MODEL_NAME = "l3cube-pune/indic-sentence-similarity-sbert"

MEMORIES_FILE = "memories_gold.json"
PHOTOS_FILE = "labelled_photos.json"

EMB_CACHE_DIR = "emb_cache"
MEM_EMB_CACHE = os.path.join(EMB_CACHE_DIR, "memories_embs.npy")
PHOTO_EMB_CACHE = os.path.join(EMB_CACHE_DIR, "photos_embs.npy")

DB_PATH = "chroma_shared_db"
BATCH_SIZE = 64
TOP_K = 6
WIPE_COLLECTIONS_BEFORE_INDEX = True  # set False to keep existing collections
# ----------------------------

os.makedirs(EMB_CACHE_DIR, exist_ok=True)

# ---------- imports ----------
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import warnings

# ---------- helpers ----------
def stable_id_from_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def make_ids_unique(ids_list):
    seen = {}
    out_ids = []
    for orig in ids_list:
        if orig not in seen:
            seen[orig] = 0
            out_ids.append(orig)
        else:
            seen[orig] += 1
            new_id = f"{orig}__{seen[orig]}"
            while new_id in seen:
                seen[orig] += 1
                new_id = f"{orig}__{seen[orig]}"
            seen[new_id] = 0
            out_ids.append(new_id)
    return out_ids

def sanitize_metadata_dict(md: dict):
    out = {}
    for k, v in (md or {}).items():
        if k == "tags":
            continue
        if v is None:
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = ", ".join(str(x) for x in v)
        else:
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[k] = str(v)
    return out

# ---------- robust Chroma client ----------
def create_chroma_client(persist_directory: str = "chroma_db"):
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        client = chromadb.Client(settings)
        return client
    except Exception:
        warnings.warn("Client(Settings) failed; trying PersistentClient...", UserWarning)
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        return client
    except Exception:
        warnings.warn("PersistentClient failed; trying Client()...", UserWarning)
    try:
        client = chromadb.Client()
        return client
    except Exception as e:
        raise RuntimeError("Failed to construct a chromadb client.") from e

# ---------- load model ----------
model = SentenceTransformer(MODEL_NAME)

# ---------- load inputs ----------
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

memories = load_json(MEMORIES_FILE)
photos = load_json(PHOTOS_FILE)

# ---------- prepare text lists ----------
mem_texts = []
mem_ids = []
mem_metadatas = []
mem_docs = []

for item in memories:
    if isinstance(item, dict):
        text = (item.get("memory") or item.get("memory_text") or "").strip()
        if not text:
            text = json.dumps(item, ensure_ascii=False)
        meta = {k: v for k, v in item.items() if k not in ("memory", "memory_text")}
    else:
        text = str(item).strip()
        meta = {}
    mem_texts.append(text)
    mem_ids.append(stable_id_from_text(text + "__mem"))
    mem_metadatas.append(sanitize_metadata_dict(meta))
    mem_docs.append(text if len(text) <= 300 else text[:297] + "...")

photo_texts = []
photo_ids = []
photo_metadatas = []
photo_docs = []

for p in photos:
    fname = p.get("filename") or p.get("id") or ""
    fp = p.get("filepath") or ""
    backstory = (p.get("backstory") or "").strip()
    if not backstory or backstory.upper().startswith("WRITE"):
        backstory = fname or fp or "photo"
    photo_texts.append(backstory)
    pid_source = os.path.abspath(fp) if fp else fname
    photo_ids.append(stable_id_from_text(pid_source + "__photo"))
    meta = {
        "filename": fname,
        "filepath": fp,
        "exif_date": p.get("exif_date", ""),
        "backstory": backstory
    }
    photo_metadatas.append(sanitize_metadata_dict(meta))
    photo_docs.append(fname)

# ---------- compute or load embeddings ----------
def compute_embeddings(texts, cache_path, batch_size=BATCH_SIZE):
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)
            if arr.shape[0] == len(texts):
                return arr
        except Exception:
            pass
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(emb)
    emb_arr = np.vstack(all_embs) if all_embs else np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    emb_arr = l2_normalize(emb_arr)
    np.save(cache_path, emb_arr)
    return emb_arr

mem_embs = compute_embeddings(mem_texts, MEM_EMB_CACHE) if mem_texts else np.zeros((0, model.get_sentence_embedding_dimension()))
photo_embs = compute_embeddings(photo_texts, PHOTO_EMB_CACHE) if photo_texts else np.zeros((0, model.get_sentence_embedding_dimension()))

# ---------- insert into Chroma ----------
client = create_chroma_client(DB_PATH)

if WIPE_COLLECTIONS_BEFORE_INDEX:
    try:
        client.delete_collection("memories")
    except Exception:
        pass
    try:
        client.delete_collection("photos_backstory")
    except Exception:
        pass

mem_collection = client.get_or_create_collection("memories")
photo_collection = client.get_or_create_collection("photos_backstory")

mem_ids_unique = make_ids_unique(mem_ids)
photo_ids_unique = make_ids_unique(photo_ids)

def add_in_chunks(col, ids_list, docs_list, metas_list, emb_array, chunk=128):
    total = len(ids_list)
    for i in range(0, total, chunk):
        j = min(total, i+chunk)
        chunk_ids = ids_list[i:j]
        chunk_docs = docs_list[i:j]
        chunk_metas = metas_list[i:j]
        chunk_embs = emb_array[i:j].tolist() if (emb_array is not None and getattr(emb_array, "size", 0)) else None
        try:
            if chunk_embs is not None:
                col.add(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas, embeddings=chunk_embs)
            else:
                col.add(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas)
        except Exception:
            try:
                col.upsert(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas, embeddings=chunk_embs)
            except Exception:
                try:
                    col.add(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas)
                except Exception as e:
                    print("Failed to add chunk:", e)

if mem_ids_unique:
    add_in_chunks(mem_collection, mem_ids_unique, mem_docs, mem_metadatas, mem_embs)
if photo_ids_unique:
    add_in_chunks(photo_collection, photo_ids_unique, photo_docs, photo_metadatas, photo_embs)

try:
    client.persist()
except Exception:
    pass

print("Done. Indexed:", len(mem_ids_unique), "memories and", len(photo_ids_unique), "photo backstories.")


# !/usr/bin/env python3
# reindex_normalize_build_db.py
# """
# Re-index script:
#  - Loads memories_gold.json and photos_backstories_template.json
#  - Normalizes date metadata into ISO / month / month_name / year
#  - Embeds texts using a Hinglish-capable model (configurable)
#  - Ensures unique IDs and sanitized metadata (no lists)
#  - Wipes and rebuilds Chroma collections: 'memories' and 'photos_backstory'

# Edit CONFIG below for paths / model names and run:
#   python reindex_normalize_build_db.py
# """

# import os, json, hashlib, warnings
# from pathlib import Path
# from datetime import datetime
# import numpy as np

# # ---------- CONFIG ----------
# EMBED_MODEL = "l3cube-pune/indic-sentence-similarity-sbert"  # Hinglish-capable; change if you prefer
# MEMORIES_FILE = "memories_gold.json"
# PHOTOS_FILE = "photos_backstories_template.json"
# EMB_CACHE_DIR = "emb_cache"
# MEM_EMB_CACHE = os.path.join(EMB_CACHE_DIR, "memories_embs.npy")
# PHOTO_EMB_CACHE = os.path.join(EMB_CACHE_DIR, "photos_embs.npy")
# CHROMA_DIR = "chroma_shared_db"
# BATCH_SIZE = 64
# WIPE_BEFORE_INDEX = True
# # --------------------------

# os.makedirs(EMB_CACHE_DIR, exist_ok=True)

# # ---------- imports ----------
# try:
#     from sentence_transformers import SentenceTransformer
# except Exception as e:
#     raise RuntimeError("Install sentence-transformers (pip install sentence-transformers torch).") from e

# try:
#     import chromadb
#     from chromadb.config import Settings
# except Exception as e:
#     raise RuntimeError("Install chromadb (pip install chromadb).") from e

# try:
#     import dateparser
# except Exception:
#     raise RuntimeError("Install dateparser (pip install dateparser).")

# # ---------- helpers ----------
# def stable_id_from_text(s: str) -> str:
#     return hashlib.md5(s.encode("utf-8")).hexdigest()

# def l2_normalize(x: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
#     return x / norms

# def sanitize_metadata(md: dict):
#     out = {}
#     for k, v in (md or {}).items():
#         if v is None:
#             out[k] = None
#         elif isinstance(v, (str, int, float, bool)):
#             out[k] = v
#         elif isinstance(v, list):
#             out[k] = ", ".join(str(x) for x in v)
#         elif isinstance(v, dict):
#             try:
#                 out[k] = json.dumps(v, ensure_ascii=False)
#             except:
#                 out[k] = str(v)
#         else:
#             out[k] = str(v)
#     return out

# def parse_date_string(s):
#     if not s: return None
#     # dateparser handles many noisy formats and relative dates
#     try:
#         dt = dateparser.parse(s, settings={'PREFER_DATES_FROM': 'past'})
#     except Exception:
#         dt = None
#     if not dt:
#         # try explicit formats
#         for fmt in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d","%d-%m-%Y","%d %B %Y","%B %d, %Y"):
#             try:
#                 dt = datetime.strptime(s, fmt); break
#             except Exception:
#                 continue
#     if not dt:
#         return None
#     return {"date_iso": dt.strftime("%Y-%m-%d"), "month": dt.month, "month_name": dt.strftime("%B").lower(), "year": dt.year}

# # ---------- robust chroma client ----------
# def create_chroma_client(persist_directory: str = CHROMA_DIR):
#     try:
#         settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
#         client = chromadb.Client(settings)
#         print("Chroma client created via Client(Settings(...))")
#         return client
#     except Exception:
#         warnings.warn("Client(Settings) failed; trying PersistentClient...", UserWarning)
#     try:
#         client = chromadb.PersistentClient(path=persist_directory)
#         print("Chroma client created via PersistentClient")
#         return client
#     except Exception:
#         warnings.warn("PersistentClient failed; trying Client()...", UserWarning)
#     client = chromadb.Client()
#     print("Chroma client created via Client()")
#     return client

# # ---------- load inputs ----------
# def load_json_or_empty(path):
#     if not os.path.exists(path):
#         print(f"Warning: {path} not found; continuing with empty list.")
#         return []
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# memories = load_json_or_empty(MEMORIES_FILE)
# photos = load_json_or_empty(PHOTOS_FILE)

# # ---------- normalize dates in memory/photo metadata ----------
# def enrich_memory_meta(item):
#     # item could be dict with 'date' or 'date_iso' etc.
#     meta = {}
#     # copy everything except raw large text fields
#     for k, v in (item.items() if isinstance(item, dict) else []):
#         meta[k] = v
#     # attempt date parse from common fields
#     raw_date_sources = []
#     if isinstance(item, dict):
#         for f in ("date", "date_iso", "timestamp", "time", "datetime"):
#             if f in item and item.get(f):
#                 raw_date_sources.append(str(item.get(f)))
#     # fallback: try to extract date from memory text with simple regex (rare)
#     # pick first candidate
#     parsed = None
#     for s in raw_date_sources:
#         parsed = parse_date_string(s)
#         if parsed:
#             break
#     if not parsed:
#         # no date sources, leave as-is
#         return meta
#     meta.update(parsed)
#     return meta

# def enrich_photo_meta(p):
#     meta = {}
#     if isinstance(p, dict):
#         meta.update(p)
#     # check exif_date fields
#     parsed = None
#     for f in ("exif_date","exifDate","date","date_iso"):
#         if f in p and p.get(f):
#             parsed = parse_date_string(str(p.get(f)))
#             if parsed: break
#     if not parsed:
#         # no exif parse
#         return meta
#     meta.update(parsed)
#     return meta

# # ---------- compute embeddings ----------
# print("Loading embedder:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)
# dim = embedder.get_sentence_embedding_dimension()
# print("Embedder dimension:", dim)

# # prepare lists
# mem_texts, mem_ids, mem_metas, mem_docs = [], [], [], []
# for i, it in enumerate(memories):
#     if isinstance(it, dict):
#         text = (it.get("memory") or it.get("memory_text") or json.dumps(it, ensure_ascii=False)).strip()
#         meta = {k: v for k, v in it.items() if k not in ("memory","memory_text")}
#     else:
#         text = str(it).strip()
#         meta = {}
#     # enrich meta with normalized date if possible
#     parsed = None
#     if isinstance(it, dict):
#         parsed = enrich_memory_meta(it)
#     if parsed:
#         meta.update(parsed)
#     mem_texts.append(text)
#     mem_ids.append(stable_id_from_text(text + "__mem"))
#     mem_metas.append(sanitize_metadata(meta))
#     mem_docs.append(text if len(text) <= 300 else text[:297] + "...")

# photo_texts, photo_ids, photo_metas, photo_docs = [], [], [], []
# for p in photos:
#     filename = p.get("filename") or p.get("id") or ""
#     filepath = p.get("filepath") or p.get("path") or ""
#     backstory = (p.get("backstory") or "").strip()
#     if not backstory or backstory.upper().startswith("WRITE"):
#         backstory = filename or filepath or "photo"
#     meta = enrich_photo_meta(p)
#     # remove 'tags' if present
#     meta.pop("tags", None)
#     photo_texts.append(backstory)
#     pid_source = os.path.abspath(filepath) if filepath else filename
#     photo_ids.append(stable_id_from_text(pid_source + "__photo"))
#     photo_metas.append(sanitize_metadata(meta))
#     photo_docs.append(filename)

# print(f"Prepared {len(mem_texts)} memories and {len(photo_texts)} photos for embedding.")

# def compute_and_cache(texts, cache_path):
#     if os.path.exists(cache_path):
#         try:
#             arr = np.load(cache_path)
#             if arr.shape[0] == len(texts):
#                 print(f"Loaded cached embeddings from {cache_path}")
#                 return arr
#             else:
#                 print("Cached embeddings mismatch, recomputing.")
#         except Exception:
#             print("Failed to load cache; recomputing.")
#     all_emb = []
#     for i in range(0, len(texts), BATCH_SIZE):
#         batch = texts[i:i+BATCH_SIZE]
#         emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=True)
#         all_emb.append(emb)
#     emb_arr = np.vstack(all_emb) if all_emb else np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
#     emb_arr = l2_normalize(emb_arr)
#     np.save(cache_path, emb_arr)
#     print("Saved embeddings to", cache_path)
#     return emb_arr

# mem_embs = compute_and_cache(mem_texts, MEM_EMB_CACHE) if mem_texts else np.zeros((0, dim))
# photo_embs = compute_and_cache(photo_texts, PHOTO_EMB_CACHE) if photo_texts else np.zeros((0, dim))

# # ---------- build chorma DB ----------
# client = create_chroma_client(CHROMA_DIR)

# if WIPE_BEFORE_INDEX:
#     try:
#         client.delete_collection("memories")
#         print("Deleted existing 'memories' collection")
#     except Exception:
#         pass
#     try:
#         client.delete_collection("photos_backstory")
#         print("Deleted existing 'photos_backstory' collection")
#     except Exception:
#         pass

# mem_col = client.get_or_create_collection("memories")
# photo_col = client.get_or_create_collection("photos_backstory")

# # make ids unique utility
# def make_ids_unique(ids):
#     seen = {}
#     out=[]
#     for orig in ids:
#         if orig not in seen:
#             seen[orig]=0
#             out.append(orig)
#         else:
#             seen[orig]+=1
#             new = f"{orig}__{seen[orig]}"
#             while new in seen:
#                 seen[orig]+=1
#                 new = f"{orig}__{seen[orig]}"
#             seen[new]=0
#             out.append(new)
#     return out

# mem_ids_unique = make_ids_unique(mem_ids)
# photo_ids_unique = make_ids_unique(photo_ids)

# def add_in_chunks(col, ids_list, docs_list, metas_list, emb_array, chunk=256):
#     total = len(ids_list)
#     for i in range(0, total, chunk):
#         j = min(total, i+chunk)
#         cids = ids_list[i:j]
#         cdocs = docs_list[i:j]
#         cmetas = metas_list[i:j]
#         cembs = emb_array[i:j].tolist() if (emb_array is not None and getattr(emb_array,"size",0)) else None
#         try:
#             if cembs is not None:
#                 col.add(ids=cids, documents=cdocs, metadatas=cmetas, embeddings=cembs)
#             else:
#                 col.add(ids=cids, documents=cdocs, metadatas=cmetas)
#         except Exception:
#             try:
#                 col.upsert(ids=cids, documents=cdocs, metadatas=cmetas, embeddings=cembs)
#             except Exception as e:
#                 print("Failed chunk:", e)

# print("Indexing memories...")
# if mem_ids_unique:
#     add_in_chunks(mem_col, mem_ids_unique, mem_docs, mem_metas, mem_embs, chunk=128)
# print("Indexing photos...")
# if photo_ids_unique:
#     add_in_chunks(photo_col, photo_ids_unique, photo_docs, photo_metas, photo_embs, chunk=128)

# try:
#     client.persist()
# except Exception:
#     pass

# print("✅ Reindexing done. Memories:", len(mem_ids_unique), "Photos:", len(photo_ids_unique))
