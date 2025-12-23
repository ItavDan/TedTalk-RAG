# Imports
import os
from dotenv import load_dotenv
load_dotenv()

# Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.llmod.ai"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is missing in .env file")
if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY is missing in .env file")

# Models
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
LLM_MODEL = "RPRTHPB-gpt-5-mini"

# Pinecone index name
PINECONE_INDEX_NAME = "ted-rag"

# RAG Parameters
RAG_CONFIG = {
    "chunk_size": 4000,
    "overlap_ratio": 0.1,
    "top_k": 3
}

# Prompts
GENERAL_PROMPT = """
You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don’t know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.
"""
CONTEXT_PROMPT = """
Given the following context, please answer the user's question:\n\n{context}
"""

# Data directories
DATA_DIR = "ted_talks_en.csv"

# Data Frame columns
TEXT_COLUMN = "transcript"
