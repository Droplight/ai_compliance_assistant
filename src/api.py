from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Now import your modules
from ingest import load_pdfs
from rag import chunk_text, build_vectorstore
from agents import run_pipeline

# -------- SETUP ONLY ONCE -------- #
print("🔄 Starting API initialization...")
print("📂 Current working directory:", os.getcwd())
print("📂 Script location:", os.path.dirname(__file__))

print("🔄 Loading PDFs...")
text_data = load_pdfs()

if len(text_data) == 0:
    print("⚠️  WARNING: No PDF data loaded. API will still start but responses may be limited.")
    chunks = []
    vectorstore = None
else:
    print("✂️  Chunking text...")
    chunks = chunk_text(text_data)
    
    print("🗄️  Building vector database...")
    vectorstore = build_vectorstore(chunks)

print("✅ API Ready!")

# -------- FASTAPI APP -------- #
app = FastAPI(
    title="AI Compliance Assistant API",
    description="Multi-agent RAG system for compliance analysis",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    """
    Health check endpoint
    """
    return {
        "status": "API is running",
        "message": "Use POST /analyze to query the system",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    """
    Detailed health check
    """
    return {
        "status": "healthy",
        "pdf_loaded": len(text_data) > 0,
        "chunks_created": len(chunks) if chunks else 0,
        "vectorstore_ready": vectorstore is not None
    }

@app.post("/analyze")
def analyze(request: QueryRequest):
    """
    Takes a 'query' and runs the multi-agent pipeline.
    Returns: retrieved context, risk analysis, and PM outputs
    """
    if vectorstore is None:
        return {
            "error": "Vector store not initialized. Please check if PDFs are loaded correctly."
        }
    
    result = run_pipeline(request.query, vectorstore)
    return result
```
