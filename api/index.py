# Imports
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .Constants import *

# Initialize FastAPI
tags_metadata = [
    {
        "name": "Prompting",
        "description": "Operations related to sending questions to the model.",
    },
    {
        "name": "System",
        "description": "Status checks and system information.",
    },
]
app = FastAPI(
    title="TED-Talk RAG Assistant API",
    description="""
    ## TED-Talk RAG Assistant API

    A Retrieval-Augmented Generation (RAG) system that answers questions based on TED Talk transcripts.

    ### Overview
    This API uses a vector database (Pinecone) to retrieve relevant TED Talk transcript chunks and an LLM to generate 
    accurate answers based solely on the provided context. The system ensures responses are grounded in actual TED Talk 
    content rather than hallucinated information.

    ### Architecture
    - **Embeddings**: OpenAI text-embedding-3-small model for semantic search
    - **LLM**: GPT-5-mini for answer generation
    - **Vector Store**: Pinecone for efficient similarity search
    - **RAG Configuration**: Configurable chunk size, overlap ratio, and top-k retrieval

    ### Main Endpoints
    - `POST /api/prompt` - Submit a question and receive an answer with context and augmented prompt
    - `GET /api/stats` - View current RAG configuration parameters
    - `GET /` - Health check endpoint

    ### Usage
    Send a POST request to `/api/prompt` with a JSON body containing your question:
    ```json
    {
        "question": "I’m looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
    }
    ```

    The API returns the answer, relevant context chunks with metadata (title, speaker, content), 
    and the augmented prompt used for generation.
    
    """,
    version="1.0.0",
    contact={
        "name": "Itav Dan",
        "email": "itavdan@gmail.com",
    },
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


# Initialize LangChain models
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=1
)
embedder = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedder,
    pinecone_api_key=PINECONE_API_KEY
)


# Initialize LLM prompt and chain
prompt_template = ChatPromptTemplate.from_messages([
    ("system", GENERAL_PROMPT),
    ("system", CONTEXT_PROMPT),
    ("human", "{input}"),
])
simple_rag_chain = prompt_template | llm | StrOutputParser()


# Set up API endpoint

# Initialize Pydantic models for request and response payloads
class QueryRequest(BaseModel):
    question: str

class ContextItem(BaseModel):
    talk_id: str
    title: str
    chunk: str
    score: Optional[float] = 0.0

class AugmentedPrompt(BaseModel):
    System: str
    User: str

class QueryResponse(BaseModel):
    response: str
    context: List[ContextItem]
    Augmented_prompt: AugmentedPrompt

# Initialize POST endpoint
@app.post("/api/prompt", response_model=QueryResponse, tags=["Prompting"])
async def handle_prompt(request: QueryRequest):
    try:
        # Retrieve top-k most similar documents from Pinecone
        results_with_scores = vectorstore.similarity_search_with_score(
            request.question,
            k=RAG_CONFIG["top_k"]
        )

        # Process results to build response and reconstructed prompt
        context_items, context_text_for_llm = [], []
        for doc, score in results_with_scores:
            title = doc.metadata.get("title", "Unknown Title")
            speaker = doc.metadata.get("speaker_1", "Unknown Speaker")
            talk_id = str(doc.metadata.get("talk_id", "N/A"))

            item = ContextItem(
                talk_id=talk_id,
                title=title,
                chunk=doc.page_content,
                score=score
            )
            context_items.append(item)

            formatted_text = (
                f"=== START OF CHUNK ===\n"
                f"Title: {title}\n"
                f"Speaker: {speaker}\n"
                f"Content:\n{doc.page_content}\n"
                f"=== END OF CHUNK ==="
            )
            context_text_for_llm.append(formatted_text)

        # Get response from LLM based on context
        full_context_string = "\n\n---\n\n".join(context_text_for_llm)
        answer = simple_rag_chain.invoke({
            "input": request.question,
            "context": full_context_string
        })

        # Final Response Payload
        final_system_prompt = f"{GENERAL_PROMPT}\n\n{CONTEXT_PROMPT.format(context=full_context_string)}"
        response_payload = QueryResponse(
            response=answer,
            context=context_items,
            Augmented_prompt=AugmentedPrompt(
                System=final_system_prompt,
                User=request.question
            )
        )

        return response_payload

    except Exception as e:
        # Log error for debugging (visible in Vercel logs)
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize GET endpoint
@app.get("/api/stats", tags=["System"])
async def get_stats():
    return RAG_CONFIG

# Initialize status endpoint
@app.get("/", tags=["System"])
async def root():
    return {"message": "TED Talk RAG Assistant is running!"}