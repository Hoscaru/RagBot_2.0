import os

# FastAPI Isntance
from fastapi import FastAPI

app = FastAPI()

# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Test Endpoint

@app.get("/")
async def test():
    return {"message": "Hello, World!"}

# LLM models instance
from langchain_cohere import ChatCohere

llm = ChatCohere()

# Implementing memory
from langgraph.checkpoint.memory import InMemorySaver

checkpoint_saver = InMemorySaver()

# Implementing tools(Tavily)
from langchain_tavily import TavilySearch

search = TavilySearch(max_results=5, topic="general")

# Create Agent react and tools
from langgraph.prebuilt import create_react_agent

tools = [search]

agent = create_react_agent(model=llm, tools=tools, checkpointer=checkpoint_saver)

# Embeddings
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Vector Store
from langchain_chroma import Chroma

vector_store = Chroma(embedding_function=embeddings, persist_directory="vector_store")

# BaseModel
from pydantic import BaseModel

class LLMRequest(BaseModel):
    promt: str
    id: int

# LLM Endpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@app.post("/llm")
async def llm_endpoint(request: LLMRequest):
    config = {"configurable": {"thread_id": request.id}}
    user_prompt = HumanMessage(content=request.promt)
    response = agent.invoke({"messages":user_prompt}, config=config)
    return {"response": response["messages"][-1].content}


from fastapi import UploadFile, File
import tempfile
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@app.post("/load_pdf")
async def load_pdf_endpoint(file: UploadFile = File(...)):
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Here you can process the PDF file as needed
        loader = PyPDFLoader(tmp.name)
        pages = []
        for page in loader.load():
            pages.append(page.page_content)
        
        splitted_pages = text_splitter.create_documents(pages)
        # Add documents to the vector store
        vector_store.add_documents(splitted_pages)
        return {"message": "PDF loaded and processed successfully.",
                "num_pages": len(splitted_pages),
                #data
                "data": splitted_pages}
    finally:
        # Clean up the temporary file
        os.remove(tmp.name)