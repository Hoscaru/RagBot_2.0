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
