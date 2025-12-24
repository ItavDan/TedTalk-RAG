# Imports
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from .Constants import *

# Initialize FastAPI
app = FastAPI()

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
        # Run RAG chain (Retrieval)
        result = rag_chain.invoke({"input": request.question})

        # Process results to build response and reconstructed prompt
        context_items, context_text_for_llm = [], []
        for doc in result["context"]:
            title = doc.metadata.get("title", "Unknown Title")
            speaker = doc.metadata.get("speaker_1", "Unknown Speaker")
            item = ContextItem(
                talk_id=str(doc.metadata.get("talk_id", "N/A")),
                title=title,
                chunk=doc.page_content,
                score=0.0
            )
            context_items.append(item)

            # --- CRITICAL FIX: Add Metadata to the text the LLM sees ---
            # This ensures the LLM knows the speaker and title for each text chunk
            formatted_text = f"Title: {title}\nSpeaker: {speaker}\nContent: {doc.page_content}"
            context_text_for_llm.append(formatted_text)

        # Full context
        full_context_string = "\n\n---\n\n".join(context_text_for_llm)

        # Format the template, Assuming CONTEXT_PROMPT contains "{context}"
        final_system_prompt = f"{GENERAL_PROMPT}\n\n{CONTEXT_PROMPT.format(context=full_context_string)}"

        # Final Response Payload
        response_payload = QueryResponse(
            response=result["answer"],
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
@app.get("/api/stats")
async def get_stats():
    return RAG_CONFIG

# Initialize status endpoint
@app.get("/")
async def root():
    return {"message": "TED Talk RAG Assistant is running!"}