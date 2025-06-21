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

# Create Agent react
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=[])

# BaseModel
from pydantic import BaseModel

class LLMRequest(BaseModel):
    promt: str

# LLM Endpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@app.post("/llm")
async def llm_endpoint(request: LLMRequest):
    user_prompt = HumanMessage(content=request.promt)
    response = agent.invoke({"messages":user_prompt})
    return {"response": response["messages"][-1].content}
