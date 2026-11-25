# Naive RAG with FAISS

A minimal **Naive (Standard) Retrieval-Augmented Generation** system that answers questions about a small collection of text documents (AI / ML / data-analytics notes in `data/`).

## Approach
1. **Load** `.txt` files from `data/`.
2. **Chunk** each document into 120-word passages with a 30-word overlap.
3. **Embed** chunks using `sentence-transformers/all-MiniLM-L6-v2` (normalised vectors).
4. **Index** vectors in a FAISS `IndexFlatIP` (inner product = cosine similarity). Index is saved to `index/`.
5. **Retrieve** the top-k chunks for the user question.
6. **Generate** an answer with an LLM that is told to use *only* the retrieved context.
   - Default: free local model `google/flan-t5-base`.
   - Optional: OpenAI (`gpt-4o-mini`) if `OPENAI_API_KEY` is set.

```
Question -> Embed -> FAISS top-k -> Prompt(context + question) -> LLM -> Answer
```

## Dependencies
Python 3.9+ and the packages in `requirements.txt` (sentence-transformers, faiss-cpu, transformers, torch, numpy).

## Setup
```bash
git clone <this-repo-url>
cd naive-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# Build index and ask one question
python src/naive_rag.py --rebuild --question "What is overfitting and how can it be reduced?"

# Interactive mode
python src/naive_rag.py

# Use OpenAI instead of the local model (optional)
export OPENAI_API_KEY=sk-...     # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
python src/naive_rag.py --question "What is drill-through in Power BI?"
```
Use `--k 5` to change how many chunks are retrieved. Add your own `.txt` files to `data/` and run with `--rebuild`.

## Sample questions
- What is the difference between supervised and unsupervised learning?
- Why did Transformers replace RNNs?
- What does FAISS do?
- What is a star schema?

## Example output
```
[load]   5 documents from 'data'
[chunk]  ~20 chunks created
[index]  FAISS index with ~20 vectors (dim=384) saved to 'index/'
[retrieve] top-3 chunks:
   score=0.6xx  machine_learning.txt (chunk 1)
   ...
ANSWER: ...
```
(Paste your own real run output / screenshot here before submitting.)

## Limitations
Naive RAG has no query rewriting or re-ranking and uses dense retrieval only, so exact keyword matches can be missed. See the companion **Hybrid RAG** repository for an improved approach.
