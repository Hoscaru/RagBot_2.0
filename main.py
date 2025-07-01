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
from langchain_core.prompts import ChatPromptTemplate

tools = [search]

agent = create_react_agent(model=llm,
                        tools=tools,
                        checkpointer=checkpoint_saver,)

# Embeddings
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Vector Store
from langchain_chroma import Chroma

vector_store = Chroma(embedding_function=embeddings, persist_directory="vector_store")

# BaseModel
from pydantic import BaseModel

class LLMRequest(BaseModel):
    prompt: str
    id: int

# LLM Endpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from fastapi import HTTPException
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

@app.post("/llm")

async def llm_endpoint(request: LLMRequest):
    try:
        # Primero verificamos si hay información relevante en el vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(request.prompt)
        
        # Construimos el contexto a partir de los documentos recuperados
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Creamos un prompt que combine el RAG con las capacidades del agente
        rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """Eres un asistente útil que combina información recuperada con conocimiento general. 
                Utiliza la siguiente información de contexto para responder a la pregunta del usuario.
                Si no sabes la respuesta o la información del contexto no es relevante, usa tus herramientas para buscar información.
                
                Contexto: {context}"""),
                ("human", "{question}")
            ])
        
        # Configuramos la cadena RAG
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | rag_prompt 
            | llm
            | StrOutputParser()
        )
        
        # Determinamos si usar RAG o el agente con herramientas
        if relevant_docs:
            # Usamos RAG si encontramos documentos relevantes
            response = rag_chain.invoke(request.prompt)
        else:
            # Usamos el agente con herramientas si no hay documentos relevantes
            response = agent.invoke({"messages": [HumanMessage(content=request.prompt)]})
        
        return {
            "response": response,
            "id": request.id,
            "relevant_docs_found": len(relevant_docs) > 0,
            "num_relevant_docs": len(relevant_docs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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