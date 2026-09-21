import json
import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """ schema for source used by agent"""
    url:str = Field(description="url of the source")

class AgentResponse(BaseModel):
    """ schema for agent response wuth answers and sources"""
    answer:str = Field(description="agents answer to query")
    sources: List[Source]= Field(default_factory=list, description="list of sources used by agent to answer query")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"), temperature=0)
search = TavilySearch(max_results=3)

# no response_format here: on ollama, forcing a schema turns off tool calling,
# so the agent would stop searching and answer from memory instead
agent = create_agent(llm, tools=[search])

# second pass, structures the agent's free-text answer
formatter = llm.with_structured_output(AgentResponse)


def collect_urls(messages) -> List[str]:
    """ pull the real urls out of the tavily tool results"""
    urls = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in payload.get("results", []):
            if item.get("url"):
                urls.append(item["url"])
    return urls


def main():
    query = "What is the weather in Tokyo?"
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    answer_text = response["messages"][-1].content
    urls = collect_urls(response["messages"])

    result: AgentResponse = formatter.invoke(
        [
            HumanMessage(
                content=(
                    "Reformat this research result into the required schema.\n\n"
                    f"Question: {query}\n\n"
                    f"Answer: {answer_text}\n\n"
                    f"Sources - use these urls verbatim, do not invent any: {urls}"
                )
            )
        ]
    )

    print(result.answer)
    for source in result.sources:
        print(f"- {source.url}")


if __name__ == "__main__":
    main()
