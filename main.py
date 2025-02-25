from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import pickle

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

# Function to load or create vector store
def get_vectorstore():
    vector_store_path = 'vectorstore.pkl'
    
    if os.path.exists(vector_store_path):
        with open(vector_store_path, 'rb') as f:
            return pickle.load(f)
    
    # Load and process the website content
    with open('website_content2.txt', 'r', encoding='utf-8') as file:
        raw_text = file.read()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size
        chunk_overlap=50,  # Reduced overlap
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    # Create embeddings with minimal model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Save vector store
    with open(vector_store_path, 'wb') as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

# Initialize conversation memory with limited history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=3  # Keep only last 3 exchanges
)

# Create conversation chain with Groq
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

# Initialize vectorstore
vectorstore = get_vectorstore()

# Create QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # Reduced number of results
    memory=memory,
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = qa_chain({"question": request.question})
        return ChatResponse(answer=response['answer'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000"))) 