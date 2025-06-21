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

# BaseModel
from pydantic import BaseModel

class LLMRequest(BaseModel):
    promt: str

# LLM Endpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@app.post("/llm")
async def llm_endpoint(request: LLMRequest):
    user_prompt = HumanMessage(content=request.promt)
    response = llm.invoke([user_prompt])
    return {"response": response.content}
