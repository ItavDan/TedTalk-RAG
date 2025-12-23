# Imports
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from Constants import *

# Initialize FastAPI
app = FastAPI()

# Initialize LangChain models
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
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
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

# Initialize retriever from Pinecone vectorstore
retriever = vectorstore.as_retriever(
    search_kwargs={"k": RAG_CONFIG["top_k"]}
)

# Create RAG chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


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
@app.post("/api/prompt", response_model=QueryResponse)
async def handle_prompt(request: QueryRequest):
    try:
        # Run RAG chain over the input question
        result = rag_chain.invoke({"input": request.question})

        # Retrieve context items from the result
        context_items = []
        for doc in result["context"]:
            item = ContextItem(
                talk_id=str(doc.metadata.get("talk_id", "N/A")),
                title=doc.metadata.get("title", "Unknown Title"),
                chunk=doc.page_content,
            )
            context_items.append(item)

        # Get final response
        response_payload = QueryResponse(
            response=result["answer"],
            context=context_items,
            Augmented_prompt=AugmentedPrompt(
                System=SYSTEM_PROMPT,
                User=request.question
            )
        )

        return response_payload

    except Exception as e:
        # Handle exceptions by returning a structured HTTP 500 error response and logging the error details to the terminal
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize GET endpoint
@app.get("/api/stats")
async def get_stats():
    return RAG_CONFIG

# Initialize status endpoint
@app.get("/")
async def root():
    return {"message": "TED Talk RAG Assistant is running!"}