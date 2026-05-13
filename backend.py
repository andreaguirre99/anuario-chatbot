import json
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Init ChromaDB ──────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="anuario",
    embedding_function=ef
)

# ── Load chunks if collection is empty ────────────────────────────────────────
if collection.count() == 0:
    print("Indexando chunks...")
    with open("anuario_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch]
        )
    print(f"Indexados {len(chunks)} chunks.")
else:
    print(f"ChromaDB ya tiene {collection.count()} chunks.")

# ── Anthropic client ───────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """Sos un asistente especializado en el Anuario de Estadísticas Educativas 2025 
de la UEICEE (Unidad de Evaluación Integral de la Calidad y Equidad Educativa), 
Ministerio de Educación, Ciudad Autónoma de Buenos Aires.

Respondé ÚNICAMENTE con información contenida en los fragmentos de datos que se te proveen.
Si la pregunta no puede responderse con los datos disponibles, decilo explícitamente: 
"No tengo esa información en los datos disponibles del Anuario."

Cuando cites cifras, mencioná el año y la categoría correspondiente.
Respondé en español, de manera clara y concisa."""

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(req: ChatRequest):
    # Retrieve relevant chunks
    results = collection.query(
        query_texts=[req.message],
        n_results=5
    )
    
    context = "\n\n---\n\n".join(results["documents"][0]) if results["documents"] else ""
    
    # Build messages
    messages = req.history.copy()
    
    user_content = f"""Datos relevantes del Anuario:
{context}

---
Pregunta del usuario: {req.message}"""
    
    messages.append({"role": "user", "content": user_content})
    
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    
    answer = response.content[0].text
    
    return {
        "answer": answer,
        "sources": [m.get("title", "") for m in results["metadatas"][0]] if results["metadatas"] else []
    }

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": collection.count()}

