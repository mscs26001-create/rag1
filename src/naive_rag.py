"""
Naive (Standard) RAG: chunk -> embed -> FAISS index -> top-k retrieval -> LLM answer.
"""
import argparse
import glob
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"          # free, runs locally on CPU
INDEX_DIR = "index"


# ---------------------------------------------------------------- 1. LOAD
def load_documents(folder):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    print(f"[load]   {len(docs)} documents from '{folder}'")
    return docs


# ---------------------------------------------------------------- 2. CHUNK
def chunk_text(text, size=120, overlap=30):
    """Split into word-based chunks with overlap so context is not cut mid-idea."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_chunks(docs):
    chunks = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"])):
            chunks.append({"source": d["source"], "chunk_id": i, "text": c})
    print(f"[chunk]  {len(chunks)} chunks created")
    return chunks


# ---------------------------------------------------------------- 3. EMBED + INDEX
def build_index(chunks, embedder):
    vecs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                           show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])      # inner product == cosine (normalised)
    index.add(vecs)
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"[index]  FAISS index with {index.ntotal} vectors (dim={vecs.shape[1]}) saved to '{INDEX_DIR}/'")
    return index


def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------- 4. RETRIEVE
def retrieve(query, embedder, index, chunks, k=3):
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


# ---------------------------------------------------------------- 5. GENERATE
def build_prompt(query, hits):
    context = "\n".join(h["text"] for h in hits)
    return ("Read the context and answer the question in a complete sentence.\n\n"
            f"Context: {context}\n\nQuestion: {query}")


def generate(prompt):
    """Uses an OpenAI-compatible API (Gemini/OpenRouter/OpenAI) if LLM_API_KEY is set,
    otherwise a local FLAN-T5 model."""
    if os.getenv("LLM_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("LLM_API_KEY"),
                        base_url=os.getenv("LLM_BASE_URL"))
        r = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}], temperature=0)
        return r.choices[0].message.content.strip()
    if not hasattr(generate, "model"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        generate.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        generate.model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    inputs = generate.tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = generate.model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return generate.tok.decode(out[0], skip_special_tokens=True).strip()


def answer(query, embedder, index, chunks, k=3):
    hits = retrieve(query, embedder, index, chunks, k)
    print("\n[retrieve] top-%d chunks:" % k)
    for h in hits:
        print(f"   score={h['score']:.3f}  {h['source']} (chunk {h['chunk_id']})")
    return generate(build_prompt(query, hits)), hits


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Naive RAG demo")
    ap.add_argument("--data", default="data")
    ap.add_argument("--rebuild", action="store_true", help="re-create the index")
    ap.add_argument("--question", help="single question; omit for interactive mode")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)
    if a.rebuild or not os.path.exists(f"{INDEX_DIR}/faiss.index"):
        index = build_index(build_chunks(load_documents(a.data)), embedder)
        _, chunks = load_index()
    else:
        index, chunks = load_index()
        print("[index]  loaded existing index")

    if a.question:
        ans, _ = answer(a.question, embedder, index, chunks, a.k)
        print("\nANSWER:", ans)
        return
    print("\nAsk questions (type 'exit' to quit).")
    while True:
        q = input("\nQ> ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        ans, _ = answer(q, embedder, index, chunks, a.k)
        print("\nANSWER:", ans)


if __name__ == "__main__":
    main()