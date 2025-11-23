from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

def build_vectorstore(chunks):
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="compliance_docs"
    )
    return vectorstore

def answer_question(query, vectorstore):
    # Step 1: Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    
    # Step 2: Combine retrieved text
    context = "\n\n".join([d.page_content for d in docs])
    
    # Step 3: Ask LLM to answer based on retrieved context
    prompt = f"""
    You are a compliance assistant. Answer ONLY using the information below:
    {context}
    
    Question: {query}
    """
    response = llm(prompt)
    return response