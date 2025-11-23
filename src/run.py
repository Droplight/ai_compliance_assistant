from ingest import load_pdfs
from rag import chunk_text, build_vectorstore, answer_question

text = load_pdfs()
chunks = chunk_text(text)
vs = build_vectorstore(chunks)

while True:
    q = input("Ask a question (or 'quit' to exit): ")
    if q.lower() == 'quit':
        break
    print("\nAnswer:", answer_question(q, vs), "\n")