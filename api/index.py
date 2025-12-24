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
@app.post("/api/prompt", response_model=QueryResponse)
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
@app.get("/api/stats")
async def get_stats():
    return RAG_CONFIG

# Initialize status endpoint
@app.get("/")
async def root():
    return {"message": "TED Talk RAG Assistant is running!"}