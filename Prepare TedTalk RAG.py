# Imports
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from Constants import *


def ingest_data():
    # Read CSV file
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found. Please download it first.")
        return
    print("Reading CSV file...")
    df = pd.read_csv(DATA_DIR)

    # Select first 10 talks for testing
    df = df.head(10)
    print(f"Processing {len(df)} talks (Safety mode ON)...")

    # Read data into LangChain documents
    loader = DataFrameLoader(df, page_content_column=TEXT_COLUMN)
    raw_documents = loader.load()

    # Calculate chunk size and chunk overlap
    chunk_size = RAG_CONFIG["chunk_size"]
    overlap_ratio = RAG_CONFIG["overlap_ratio"]
    chunk_overlap = int(chunk_size * overlap_ratio)

    # Split documents into chunks
    print(f"Splitting text with Chunk Size: {chunk_size}, Overlap: {chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    documents_chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(documents_chunks)} chunks from source data.")

    # Create embedder
    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    # Embed chunks and upload to Pinecone
    print(f"Uploading to Pinecone Index: {PINECONE_INDEX_NAME}...")
    PineconeVectorStore.from_documents(
        documents=documents_chunks,
        embedding=embedder,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )

    print("Ingestion complete! Data is now in Pinecone.")


if __name__ == "__main__":
    ingest_data()