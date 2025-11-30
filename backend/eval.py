#!/usr/bin/env python3
# eval_agentic_policy_only.py
"""
Purely agentic RAG: routing decided by an LLM-policy (Gemma via Ollama).
- Policy receives ONLY the user query and returns route: photo|memory|both|clarify
- Clarify triggers a one-line user prompt
- Retrieves from chosen collections and composes final JSON answer via composer LLM
- Minimal CLI output (final parsed JSON)
"""

import json, os, sys, warnings
from typing import List, Dict, Any
import numpy as np

# ---------- CONFIG ----------
# MODEL_NAME = "intfloat/e5-large-v2"  # Must match index-time encoder
MODEL_NAME = "l3cube-pune/indic-sentence-similarity-sbert"
CHROMA_DIR = "chroma_shared_db"
OLLAMA_MODEL = "gemma3:12b"         # Ollama model name
TOP_K = 6
MAX_CONTEXT_ITEMS = 12
DEBUG = True
# --------------------------------

# ---------- imports ----------
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Install sentence-transformers: pip install sentence-transformers torch") from e

try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    raise RuntimeError("Install chromadb: pip install chromadb") from e

try:
    from langchain_ollama import ChatOllama
except Exception as e:
    raise RuntimeError("Install langchain-ollama and run ollama locally with your model") from e

# ---------- init ----------
encoder = SentenceTransformer(MODEL_NAME)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

def create_chroma_client(persist_directory: str = CHROMA_DIR):
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        return chromadb.Client(settings)
    except Exception:
        try:
            return chromadb.PersistentClient(path=persist_directory)
        except Exception:
            return chromadb.Client()

client = create_chroma_client(CHROMA_DIR)
mem_col = client.get_or_create_collection("memories")
photo_col = client.get_or_create_collection("photos_backstory")

# ---------- utilities ----------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def embed_text(s: str) -> np.ndarray:
    emb = encoder.encode([s], convert_to_numpy=True)
    return l2_normalize(emb)[0]

def query_col_with_fallback(col, q_emb: np.ndarray, q_text: str, top_k: int):
    """Try embedding query, fallback to text query if embedding fails."""
    try:
        resp = col.query(query_embeddings=q_emb.reshape(1, -1).tolist(), n_results=top_k)
    except Exception:
        resp = col.query(query_texts=[q_text], n_results=top_k)
    ids = resp.get("ids", [[]])[0]
    metas = resp.get("metadatas", [[]])[0] if "metadatas" in resp else []
    docs = resp.get("documents", [[]])[0] if "documents" in resp else []
    dists = resp.get("distances", [[]])[0] if "distances" in resp else None
    return {"ids": ids, "metas": metas, "docs": docs, "dists": dists}

# ---------- LLM policy router (pure agentic) ----------
POLICY_SYSTEM = """You are a routing assistant. You receive a single user query.
Decide whether the system should search "photo", "memory", or "both", or ask the user for clarification.
Return a JSON object ONLY, with keys:
- "route": one of "photo", "memory", "both", "clarify"
- "clarify": string (only present if route == "clarify"), a single short clarification question phrase.
Be terse."""
def llm_policy_router(query: str) -> Dict[str, str]:
    messages = [
        ("system", POLICY_SYSTEM),
        ("human", f"User query: {query}\n\nDecide route.")]
    resp = llm.invoke(messages)
    txt = resp.content.strip()
    # try parsing
    try:
        start = txt.find("{")
        end = txt.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(txt[start:end])
    except Exception:
        pass
    # fallback if parsing fails
    return {"route": "both"}

# ---------- composer prompt builder ----------
COMPOSER_SYSTEM = """You are a precise assistant. Use ONLY the provided context items.
Do NOT invent facts. Cite facts using source tokens like [PHOTO:IMG_0012.jpg] or [MEM:a0f1824d].
Return a JSON object with keys:
- "answer": string,
- "sources": list of source tokens used,
- "photo_filenames": list of filenames to show.
If the answer is not in the context, return exactly: "I don't know — not found in the provided items." with empty lists for sources and photo_filenames.
"""
def construct_composer_messages(query: str, context_items: List[Dict[str, Any]]) -> List[tuple]:
    lines = []
    for it in context_items:
        if it["type"] == "photo":
            tok = f"[PHOTO:{it['id']}]"
            line = f"{tok} backstory: {it['text']}"
            if it.get("date"):
                line += f" | date: {it['date']}"
            lines.append(line)
        else:
            tok = f"[MEM:{it['id']}]"
            line = f"{tok} memory: {it['text']}"
            if it.get("date"):
                line += f" | date: {it['date']}"
            lines.append(line)
    human = f"User question: {query}\n\nContext items (use only these):\n"
    for i, l in enumerate(lines, start=1):
        human += f"{i}. {l}\n"
    human += "\nAnswer concisely and cite sources."
    return [("system", COMPOSER_SYSTEM), ("human", human)]

# ---------- assemble context items ----------
def build_context_items(photo_hits, mem_hits, max_items=MAX_CONTEXT_ITEMS):
    items = []
    for i, pid in enumerate(photo_hits["ids"]):
        meta = photo_hits["metas"][i] if i < len(photo_hits["metas"]) else {}
        text = meta.get("backstory", "") or (photo_hits["docs"][i] if i < len(photo_hits["docs"]) else "")
        date = meta.get("exif_date", "")
        filename = meta.get("filename", pid)
        items.append({"type":"photo","id":filename,"text":text,"date":date})
    for i, mid in enumerate(mem_hits["ids"]):
        meta = mem_hits["metas"][i] if i < len(mem_hits["metas"]) else {}
        text = meta.get("memory") or (mem_hits["docs"][i] if i < len(mem_hits["docs"]) else "")
        date = meta.get("date", "")
        items.append({"type":"mem","id": mid,"text": text,"date": date})
    return items[:max_items]

# ---------- main pipeline (policy-only routing) ----------
def run_agent(query: str) -> Dict[str, Any]:
    # 1) call policy LLM (only the query)
    policy = llm_policy_router(query)
    route = policy.get("route", "both")
    if DEBUG:
        print("[policy raw]", policy)

    # 2) if clarify, ask user for one-line clarification
    if route == "clarify":
        clarify_q = policy.get("clarify", "Do you want photos, chat messages, or both?")
        clar = input(f"\nClarification: {clarify_q} (type 'photo'/'memory'/'both') > ").strip().lower()
        if clar in ("photo","memory","both"):
            route = clar
        else:
            print("Unrecognized response; defaulting to 'both'.")
            route = "both"

    # 3) embed query
    q_emb = embed_text(query)

    # 4) retrieve based on route
    mem_hits = {"ids":[],"metas":[],"docs":[],"dists":None}
    photo_hits = {"ids":[],"metas":[],"docs":[],"dists":None}
    if route in ("memory","both"):
        mem_hits = query_col_with_fallback(mem_col, q_emb, query, TOP_K)
    if route in ("photo","both"):
        photo_hits = query_col_with_fallback(photo_col, q_emb, query, TOP_K)

    # 5) build context and call composer
    context_items = build_context_items(photo_hits, mem_hits, max_items=MAX_CONTEXT_ITEMS)
    messages = construct_composer_messages(query, context_items)
    comp_resp = llm.invoke(messages)
    comp_text = comp_resp.content

    # 6) try parse JSON
    parsed = None
    try:
        s = comp_text.strip()
        start = s.find("{")
        end = s.rfind("}") + 1
        if start != -1 and end != -1:
            parsed = json.loads(s[start:end])
    except Exception:
        parsed = None

    merged = {
        "query": query,
        "route_used": route,
        "context_items": context_items,
        "composer_raw": comp_text,
        "composer_json": parsed
    }
    return merged

# ---------- interactive ----------
def interactive():
    print("Agentic RAG (policy-only). Empty input to exit.")
    while True:
        q = input("\nQuery > ").strip()
        if not q:
            break
        out = run_agent(q)
        if out["composer_json"] is not None:
            print("\n=== FINAL ANSWER ===\n")
            print(json.dumps(out["composer_json"], indent=2, ensure_ascii=False))
        else:
            print("\nLLM composer did not return valid JSON. Raw output:\n")
            print(out["composer_raw"])
        if DEBUG:
            print("\n[DEBUG INFO]")
            print(json.dumps({"route_used": out["route_used"], "context_items": out["context_items"]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        interactive()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)


#!/usr/bin/env python3
# agent_probe_policy_datefilter.py
# """
# Runtime agent:
#  - Uses an LLM-policy that receives numeric retrieval signals (photo vs memory) and chooses route
#  - Detects month/year in query and filters candidates by date metadata before rerank
#  - Optional cross-encoder rerank (fast)
#  - Composer LLM (Gemma via Ollama) produces final JSON answer with sources and photo_filenames

# Edit CONFIG below. Run:
#   python agent_probe_policy_datefilter.py
# """

# import json, os, re, sys, warnings
# from typing import List, Dict, Any
# from datetime import datetime
# import numpy as np

# # ---------- CONFIG ----------
# EMBED_MODEL = "l3cube-pune/indic-sentence-similarity-sbert"  # must match reindex model you used
# CHROMA_DIR = "chroma_shared_db"
# OLLAMA_MODEL = "gemma3:12b"
# TOP_PROBE = 8      # probe depth for numeric signals
# TOP_N_RERANK = 50  # initial candidates to rerank
# TOP_K_RETURN = 6   # final top-K passed to composer
# SIM_THRESHOLD = 0.56
# MARGIN = 1.10
# MAX_CONTEXT_ITEMS = 12
# USE_RERANKER = True   # set False to skip cross-encoder rerank
# RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# DEBUG = False
# # --------------------------------

# # ---------- imports ----------
# try:
#     from sentence_transformers import SentenceTransformer
# except Exception as e:
#     raise RuntimeError("Install sentence-transformers: pip install sentence-transformers torch") from e

# try:
#     import chromadb
#     from chromadb.config import Settings
# except Exception:
#     raise RuntimeError("Install chromadb: pip install chromadb")

# try:
#     from langchain_ollama import ChatOllama
# except Exception:
#     raise RuntimeError("Install langchain-ollama and run Ollama locally with your model")

# # optional cross-encoder
# if USE_RERANKER:
#     try:
#         from sentence_transformers import CrossEncoder
#     except Exception:
#         print("Cross-encoder not available; set USE_RERANKER=False or install sentence-transformers extra.")

# # ---------- init ----------
# embedder = SentenceTransformer(EMBED_MODEL)
# embed_dim = embedder.get_sentence_embedding_dimension()
# print("Embedder dim:", embed_dim)

# if USE_RERANKER:
#     try:
#         reranker = CrossEncoder(RERANKER_NAME)
#         print("Cross-encoder loaded:", RERANKER_NAME)
#     except Exception as e:
#         print("Failed to load cross-encoder:", e)
#         USE_RERANKER = False

# client = None
# def create_chroma_client(persist_directory: str = CHROMA_DIR):
#     try:
#         settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
#         return chromadb.Client(settings)
#     except Exception:
#         try:
#             return chromadb.PersistentClient(path=persist_directory)
#         except Exception:
#             return chromadb.Client()

# client = create_chroma_client(CHROMA_DIR)
# mem_col = client.get_or_create_collection("memories")
# photo_col = client.get_or_create_collection("photos_backstory")

# policy_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
# composer_llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

# # ---------- helpers ----------
# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
#     return v / norms

# def embed_query(q: str) -> np.ndarray:
#     emb = embedder.encode([q], convert_to_numpy=True)
#     return l2_normalize(emb)[0]

# def parse_date_like(datestr: str):
#     if not datestr: return None
#     for fmt in ("%Y-%m-%d","%d/%m/%Y","%d/%m/%y","%d-%m-%Y","%d %B %Y","%B %d, %Y"):
#         try:
#             return datetime.strptime(datestr, fmt)
#         except:
#             continue
#     # last-resort extract year
#     m = re.search(r"(\d{4})", str(datestr))
#     if m:
#         try:
#             return datetime(int(m.group(1)),1,1)
#         except:
#             return None
#     return None

# def detect_month_year(query: str):
#     q = query.lower()
#     # month name
#     m = re.search(r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', q)
#     year_m = re.search(r'\b(20\d{2}|\d{4})\b', q)
#     month_idx = None
#     year_val = None
#     if m:
#         mon = m.group(1)[:3]
#         mon_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
#         month_idx = mon_map.get(mon)
#         if year_m:
#             year_val = int(year_m.group(1))
#     else:
#         # numeric patterns like 03/2025 or 3/2025
#         mm = re.search(r'(\d{1,2})[/-](\d{4})', q)
#         if mm:
#             m_i = int(mm.group(1)); y_i = int(mm.group(2))
#             if 1 <= m_i <= 12:
#                 month_idx = m_i; year_val = y_i
#     return month_idx, year_val

# # ---------- probe + policy ----------
# POLICY_SYSTEM = """You are a routing assistant. You will be given:
# - user_query: string
# - numeric signals for PHOTO and MEMORY collections: top1_sim, mean_top5, count_above_t, n_results

# Return ONLY JSON: {"route":"photo"|"memory"|"both"|"clarify", "clarify":"...?" optional }

# Follow deterministic preferences:
# - If PHOTO.top1_sim >= MEMORY.top1_sim * MARGIN and PHOTO.count_above_t >= 1 => photo
# - If MEMORY.top1_sim >= PHOTO.top1_sim * MARGIN and MEMORY.count_above_t >= 1 => memory
# - If the query is ambiguous and you need a short clarification, return {"route":"clarify","clarify":"<one-line question>"}.

# Be terse and return JSON only.
# """

# def compute_probe_signals(col, q_emb, q_text, top_probe=TOP_PROBE):
#     try:
#         resp = col.query(query_embeddings=q_emb.reshape(1,-1).tolist(), n_results=top_probe)
#     except Exception:
#         resp = col.query(query_texts=[q_text], n_results=top_probe)
#     ids = resp.get("ids", [[]])[0]
#     metas = resp.get("metadatas", [[]])[0] if "metadatas" in resp else []
#     docs = resp.get("documents", [[]])[0] if "documents" in resp else []
#     dists = resp.get("distances", [[]])[0] if "distances" in resp else None
#     sims = []
#     if dists:
#         for d in dists:
#             try:
#                 sims.append(1.0 - float(d))
#             except:
#                 sims.append(0.0)
#     else:
#         sims = [0.0]*len(ids)
#     top1_sim = float(sims[0]) if sims else 0.0
#     mean_top5 = float(np.mean(sims[:5])) if sims else 0.0
#     count_above_t = int(sum(1 for s in sims if s >= SIM_THRESHOLD))
#     return {"ids": ids, "metas": metas, "docs": docs, "sims": sims,
#             "top1_sim": top1_sim, "mean_top5": mean_top5,
#             "count_above_t": count_above_t, "n_results": len(ids)}

# def decide_route_with_probes(query, q_emb):
#     photo_probe = compute_probe_signals(photo_col, q_emb, query, TOP_PROBE)
#     mem_probe = compute_probe_signals(mem_col, q_emb, query, TOP_PROBE)

#     p_top = photo_probe["top1_sim"]; m_top = mem_probe["top1_sim"]
#     p_count = photo_probe["count_above_t"]; m_count = mem_probe["count_above_t"]

#     if p_top >= m_top * MARGIN and p_count >= 1:
#         return "photo", photo_probe, mem_probe
#     if m_top >= p_top * MARGIN and m_count >= 1:
#         return "memory", photo_probe, mem_probe

#     # ambiguous -> ask LLM policy with numeric signals
#     policy_input = (
#         f"user_query: {query}\n\n"
#         f"PHOTO: top1_sim={p_top:.3f} mean_top5={photo_probe['mean_top5']:.3f} count_above_t={p_count} n_results={photo_probe['n_results']}\n"
#         f"MEMORY: top1_sim={m_top:.3f} mean_top5={mem_probe['mean_top5']:.3f} count_above_t={m_count} n_results={mem_probe['n_results']}\n"
#         "Decide route and return JSON."
#     )
#     resp = policy_llm.invoke([("system", POLICY_SYSTEM), ("human", policy_input)])
#     t = resp.content.strip()
#     try:
#         s = t.find("{"); e = t.rfind("}")+1
#         if s!=-1 and e!=-1:
#             out = json.loads(t[s:e])
#             route = out.get("route","both")
#             if route == "clarify":
#                 return "clarify", photo_probe, mem_probe, out.get("clarify")
#             return route, photo_probe, mem_probe
#     except Exception:
#         pass
#     return "both", photo_probe, mem_probe

# # ---------- rerank helpers ----------
# def rerank_with_crossencoder(query, candidates, top_k):
#     # candidates: list of dicts {"id","text","meta","date"}
#     if not USE_RERANKER or not candidates:
#         return candidates[:top_k]
#     pairs = [(query, c["text"] + f" [DATE:{c.get('date','')}]") for c in candidates]
#     scores = reranker.predict(pairs, show_progress_bar=False)
#     for c, s in zip(candidates, scores):
#         c["_score"] = float(s)
#     out = sorted(candidates, key=lambda x: x["_score"], reverse=True)[:top_k]
#     return out

# # ---------- composer builder ----------
# COMPOSER_SYSTEM = """You are a precise assistant. Use ONLY the provided context items. Do NOT invent facts.
# Return strict JSON: {"answer": "...", "sources": ["MEM:...","PHOTO:..."], "photo_filenames":[...]}.
# If answer not found, return: {"answer":"I don't know — not found in the provided items.","sources":[],"photo_filenames":[]}"""

# def build_composer_messages(query, items):
#     lines=[]
#     for it in items:
#         if it["type"]=="photo":
#             tok=f"[PHOTO:{it['id']}]"; lines.append(f"{tok} backstory: {it['text']} | date: {it.get('date','')}")
#         else:
#             tok=f"[MEM:{it['id']}]"; lines.append(f"{tok} memory: {it['text']} | date: {it.get('date','')}")
#     human = f"User question: {query}\nContext items (use only these):\n"
#     for i,l in enumerate(lines,1):
#         human += f"{i}. {l}\n"
#     human += "\nAnswer concisely and cite sources."
#     return [("system", COMPOSER_SYSTEM), ("human", human)]

# # ---------- main pipeline ----------
# def run_agent_once(query):
#     q_emb = embed_query(query)
#     route_decision = decide_route_with_probes(query, q_emb)
#     if isinstance(route_decision, tuple) and route_decision[0]=="clarify":
#         _, photo_probe, mem_probe, clar_q = route_decision
#         clar = input(f"Clarify: {clar_q} (photo/memory/both) > ").strip().lower()
#         if clar in ("photo","memory","both"):
#             route = clar
#         else:
#             route = "both"
#         photo_probe = photo_probe; mem_probe = mem_probe
#     else:
#         route, photo_probe, mem_probe = route_decision

#     if DEBUG:
#         print("[DEBUG] route:", route)
#         print("[DEBUG] photo_probe top1:", photo_probe["top1_sim"], "mem_probe top1:", mem_probe["top1_sim"])

#     # get initial candidate pools (top_N for rerank)
#     mem_candidates=[]; photo_candidates=[]
#     if route in ("memory","both"):
#         m_resp = mem_col.query(query_embeddings=q_emb.reshape(1,-1).tolist(), n_results=TOP_N_RERANK)
#         m_ids = m_resp.get("ids",[[]])[0]; m_metas = m_resp.get("metadatas",[[]])[0]; m_docs = m_resp.get("documents",[[]])[0]
#         for i, mid in enumerate(m_ids):
#             meta = m_metas[i] if i < len(m_metas) else {}
#             doc = m_docs[i] if i < len(m_docs) else ""
#             txt = meta.get("memory") or doc
#             mem_candidates.append({"id": mid, "text": txt, "meta": meta, "date": meta.get("date") or meta.get("date_iso","")})
#     if route in ("photo","both"):
#         p_resp = photo_col.query(query_embeddings=q_emb.reshape(1,-1).tolist(), n_results=TOP_N_RERANK)
#         p_ids = p_resp.get("ids",[[]])[0]; p_metas = p_resp.get("metadatas",[[]])[0]; p_docs = p_resp.get("documents",[[]])[0]
#         for i, pid in enumerate(p_ids):
#             meta = p_metas[i] if i < len(p_metas) else {}
#             doc = p_docs[i] if i < len(p_docs) else ""
#             txt = meta.get("backstory") or doc
#             fname = meta.get("filename", pid)
#             photo_candidates.append({"id": fname, "text": txt, "meta": meta, "date": meta.get("date_iso") or meta.get("exif_date","")})

#     # date filter (if query mentions month/year)
#     month, year = detect_month_year(query)
#     if month or year:
#         if photo_candidates:
#             photo_candidates = [c for c in photo_candidates if (c.get("date") and parse_date_like(c["date"]) and ((month is None or parse_date_like(c["date"]).month==month) and (year is None or parse_date_like(c["date"]).year==year)))]
#         if mem_candidates:
#             mem_candidates = [c for c in mem_candidates if (c.get("date") and parse_date_like(c["date"]) and ((month is None or parse_date_like(c["date"]).month==month) and (year is None or parse_date_like(c["date"]).year==year)))]
#         if DEBUG:
#             print(f"[DEBUG] After date filter: {len(photo_candidates)} photos, {len(mem_candidates)} memories")

#     # rerank separately then merge
#     reranked = []
#     if photo_candidates:
#         pr = rerank_with_crossencoder(query, photo_candidates, top_k=TOP_K_RETURN)
#         for p in pr: reranked.append({"type":"photo","id":p["id"],"text":p["text"],"date":p.get("date"), "score": p.get("_score", 0.0)})
#     if mem_candidates:
#         mr = rerank_with_crossencoder(query, mem_candidates, top_k=TOP_K_RETURN)
#         for m in mr: reranked.append({"type":"mem","id":m["id"],"text":m["text"],"date":m.get("date"), "score": m.get("_score", 0.0)})

#     # sort by score and pick final top-K
#     if reranked:
#         final_sorted = sorted(reranked, key=lambda x: x.get("score",0.0), reverse=True)[:TOP_K_RETURN]
#     else:
#         final_sorted = []

#     # build composer context and call composer
#     context_items = [{"type": c["type"], "id": c["id"], "text": c["text"], "date": c.get("date","")} for c in final_sorted[:MAX_CONTEXT_ITEMS]]
#     messages = build_composer_messages(query, context_items)
#     resp = composer_llm.invoke(messages)
#     comp_text = resp.content

#     parsed = None
#     try:
#         s = comp_text.strip(); a = s.find("{"); b = s.rfind("}")+1
#         if a!=-1 and b!=-1:
#             parsed = json.loads(s[a:b])
#     except Exception:
#         parsed = None

#     return {"route": route, "probes": {"photo": photo_probe, "mem": mem_probe}, "candidates": final_sorted, "composer_raw": comp_text, "composer_json": parsed}

# # ---------- interactive ----------
# def interactive_loop():
#     print("Agent (probe+policy+datefilter). Empty input to exit.")
#     while True:
#         q = input("\nQuery > ").strip()
#         if not q: break
#         out = run_agent_once(q)
#         if out["composer_json"] is not None:
#             print("\n=== FINAL ANSWER ===\n")
#             print(json.dumps(out["composer_json"], indent=2, ensure_ascii=False))
#         else:
#             print("\nComposer raw:\n")
#             print(out["composer_raw"])
#         if DEBUG:
#             print("\n[DEBUG] route/probes/candidates:\n")
#             print(json.dumps({"route": out["route"], "probes": out["probes"], "candidates": out["candidates"]}, indent=2, ensure_ascii=False))

# if __name__ == "__main__":
#     try:
#         interactive_loop()
#     except KeyboardInterrupt:
#         print("\nExiting.")
#         sys.exit(0)
