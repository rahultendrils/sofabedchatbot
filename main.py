from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter  # Simpler splitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import gc  # For garbage collection

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# Initialize global variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize minimal embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smallest model
    model_kwargs={'device': 'cpu'}
)

# Load and process content only once at startup
with open('website_content2.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Use simpler text splitter with smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=200,  # Very small chunks
    chunk_overlap=20,
    length_function=len
)
texts = text_splitter.split_text(raw_text)

# Create vector store with minimal configuration
vectorstore = FAISS.from_texts(
    texts, 
    embeddings,
    n_lists=50  # Reduce index size
)

# Initialize Groq client with minimal config
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

# Clean up memory
del texts
del raw_text
gc.collect()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get relevant documents
        docs = vectorstore.similarity_search(
            request.question, 
            k=2  # Reduce number of results
        )
        
        # Prepare context
        context = "\n".join(doc.page_content for doc in docs)
        
        # Generate response using minimal prompt
        prompt = f"Context: {context}\nQuestion: {request.question}\nAnswer:"
        response = llm.invoke(prompt)
        
        # Clean up
        del docs
        del context
        gc.collect()
        
        return ChatResponse(answer=str(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 