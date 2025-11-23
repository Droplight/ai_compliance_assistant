from ingest import load_pdfs
from rag import chunk_text, build_vectorstore
from agents import run_pipeline

# 1. Load PDFs
text = load_pdfs()

# 2. Chunk
chunks = chunk_text(text)

# 3. Build vectorstore
vs = build_vectorstore(chunks)

# 4. Ask a question
query = "What are the compliance risks with storing customer data?"

result = run_pipeline(query, vs)

print("\nRETRIEVED CONTEXT:\n", result["retrieved_context"])
print("\nRISK ANALYSIS:\n", result["risk_analysis"])
print("\nPM OUTPUT:\n", result["pm_output"])
