# AGENT 1 — RETRIEVER
def retrieve_context(query, vectorstore):
    """
    Returns the most relevant chunks from the vector DB.
    """
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([r.page_content for r in results])

# AGENT 2 — RISK ANALYZER
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_risk(context):
    """
    Returns risk level, reasons, and flagged issues.
    """
    prompt = f"""
    You are a compliance risk expert.
    Read the following context and return:
    - Risk level (low, medium, high)
    - Reasons
    - Specific flagged issues

    Context:
    {context}
    """

    return llm(prompt)

# AGENT 3 — PRODUCT MANAGER OUTPUTS
def generate_pm_outputs(context):
    """
    Generates:
    - user stories
    - acceptance criteria
    - product recommendations
    """
    prompt = f"""
    You are a Senior Product Manager.
    Based only on the context below, generate:

    1. User stories (as: As a ___, I want ___ so that ___)
    2. Acceptance criteria (Gherkin format)
    3. Product recommendations

    Context:
    {context}
    """

    return llm(prompt)

# MULTI-AGENT PIPELINE
def run_pipeline(query, vectorstore):
    context = retrieve_context(query, vectorstore)
    risk = analyze_risk(context)
    pm_output = generate_pm_outputs(context)

    return {
        "retrieved_context": context,
        "risk_analysis": risk,
        "pm_output": pm_output
    }
