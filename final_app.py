import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Annotated
import fitz
import faiss
import numpy as np
import google.generativeai as genai

# --- Config ---
GEMINI_API_KEY = "AIzaSyDrN6pn2su6FwkPdRKPaHFZxjWuE7AjcOo"
SECRET_TOKEN = "479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
executor = ThreadPoolExecutor(max_workers=15)

# --- FastAPI Setup ---
app = FastAPI(title="HackRx LLM - Gemini Optimized")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Auth ---
async def verify_token(authorization: Annotated[str, Header()]):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth format.")
    if authorization.split(" ")[1] != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token.")

# --- Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Helpers ---
def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

def chunk_text(text: str, max_tokens: int = 300, stride: int = 150) -> List[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks, chunk, token_count = [], [], 0
    for sentence in sentences:
        tokens = sentence.split()
        if token_count + len(tokens) <= max_tokens:
            chunk.append(sentence)
            token_count += len(tokens)
        else:
            chunks.append(". ".join(chunk).strip())
            chunk = sentence.split()[stride:]
            token_count = len(chunk)
    if chunk:
        chunks.append(". ".join(chunk).strip())
    return chunks

# --- Embedding & Answering ---
def embed_chunk(text, task="retrieval_document"):
    return genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task
    )["embedding"]

def build_index(embeddings: List[List[float]]):
    dim = len(embeddings[0])
    matrix = np.array(embeddings).astype("float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index

def generate_answer(prompt: str) -> str:
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return "Information not found in the policy."

# --- Async Wrappers ---
async def embed_texts_parallel(texts: List[str], task: str) -> List[List[float]]:
    loop = asyncio.get_event_loop()
    return await asyncio.gather(*[
        loop.run_in_executor(executor, embed_chunk, text, task) for text in texts
    ])

async def generate_answers_parallel(prompts: List[str]) -> List[str]:
    loop = asyncio.get_event_loop()
    return await asyncio.gather(*[
        loop.run_in_executor(executor, generate_answer, prompt) for prompt in prompts
    ])

# --- Smart Answering with Relevance Filtering ---
async def answer_question(q: str, chunks: List[str], index: faiss.Index) -> str:
    query_embedding = (await embed_texts_parallel([q], task="retrieval_query"))[0]
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)
    scores, I = index.search(query_vector, k=10)
    relevant = [(chunks[i], scores[0][j]) for j, i in enumerate(I[0]) if scores[0][j] > 0.3]
    top_chunks = [text for text, _ in relevant]
    context = "\n".join(top_chunks)
    prompt = f"""
You are a policy analyst. Read the context below and answer the user's question clearly.

Context:
{context}

Question: {q}

Answer in one line. If not clearly found, say: 'Information not found in the policy.'
""".strip()
    reply = (await generate_answers_parallel([prompt]))[0]
    # Smart fallback: only if generic failure detected
    if "information not found" in reply.lower():
        full_prompt = f"""
Use the entire policy text below to answer.

Full Text:
{" ".join(chunks)}

Question: {q}

Answer in one clear line. Say 'Information not found in the policy' if unsure.
""".strip()
        fallback = await generate_answers_parallel([full_prompt])
        return fallback[0]
    return reply

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run(payload: HackRxRequest, token: Annotated[str, Depends(verify_token)]):
    try:
        text = extract_text_from_url(payload.documents)
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text found.")
        embeddings = await embed_texts_parallel(chunks, task="retrieval_document")
        index = build_index(embeddings)
        tasks = [answer_question(q, chunks, index) for q in payload.questions]
        answers = await asyncio.gather(*tasks)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
