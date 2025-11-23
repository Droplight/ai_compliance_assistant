from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from ingest import load_pdfs
from rag import chunk_text, build_vectorstore
from agents import run_pipeline

# -------- SETUP ONLY ONCE -------- #

print("Loading PDFs...")
text_data = load_pdfs()

print("Chunking text...")
chunks = chunk_text(text_data)

print("Building vector database...")
vectorstore = build_vectorstore(chunks)

# -------- FASTAPI APP -------- #

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/analyze")
def analyze(request: QueryRequest):
    """
    Takes a 'query' and runs the multi-agent pipeline.
    """
    result = run_pipeline(request.query, vectorstore)
    return result
