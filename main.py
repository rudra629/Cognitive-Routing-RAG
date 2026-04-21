import os 
import json
from dotenv import load_dotenv  
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

load_dotenv()

llm=ChatGroq(model="llama3-8b-8192", temperature=0.7)
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

Persona={
    "Bot A":"I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "Bot B":"I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "Bot C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI.",
}
print("\n-------------Phase 1: Vector-Based Persona Matching (The Router)-------------\n")
vector_store = Chroma(embedding_function=embeddings)

docs=[Document(page_content=text, metadata={"bot_id":name}) for name, text in Persona.items()]
def route_to_bots(post_content:str,threshold:float=0.50):
    # routes a post to relevent bots
    results=vector_store.similarity_search_with_relevance_scores(post_content, k=3)
    matched_bots=[]
    for doc, score in results:
        if score>=threshold:
            matched_bots.append((doc.metadata["bot_id"],score))
    return matched_bots

sample_post="What do you think about the future of AI and its impact on society?"
print(f"Post: {sample_post}\n")
matched=route_to_bots(sample_post)
print(f"Matched Bots: {matched}")

# phase 2
print("\n--- Phase 2: langgraph autonomous content Engine ---")

@tool 
def mock_search(query:str)->str:
    # mock search tool that returns a fixed result
    query = query.lower()
    if "crypto" in query or "bitcoin" in query:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals."
    elif "ai" in query or "model" in query:
        return "OpenAI releases new model, sparking fierce debate on the future of tech."
    elif "market" in query or "rates" in query:
        return "Federal Reserve hints at interest rate cuts in upcoming quarter."
    return "Global markets remain steady amidst tech boom."

class agentstate(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    final_post: dict

# node: 1 decides search
def decide_search(state: agentstate):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are: {persona}. Decide a topic to post about today. Output ONLY a 2-3 word search query to find news about it. No quotes, no extra text.")
    ])
    chain = prompt | llm
    res = chain.invoke({"persona": state["persona"]})
    return {"search_query": res.content.strip()}

# node 2: performs search
def web_search(state: agentstate):
    res = mock_search.invoke({"query": state["search_query"]})
    return {"search_results": res}

# node 3: creates final post with strict JSON
class PostOutput(BaseModel):
    bot_id: str = Field(description="The id of the bot")
    topic: str = Field(description="The topic of the post")
    post_content: str = Field(description="Highly opinionated post, under 280 characters")

def draft_post(state: agentstate):
    parser = JsonOutputParser(pydantic_object=PostOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are: '{persona}'. Generate a highly opinionated post (under 280 chars) based on the Context.\n\n{format_instructions}"),
        ("user", "Context: {context}")
    ])
    chain = prompt | llm | parser
    try:
        res = chain.invoke({
            "persona": state["persona"],
            "context": state["search_results"],
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        print(f"warning JSON parsing failed model output wasn't strict JSON. error: {e}")
        res = {"bot_id": "Error", "topic": "Error", "post_content": "Failed to parse."}
        
    res["bot_id"] = state["bot_id"] 
    return {"final_post": res}

workflow = StateGraph(agentstate)
workflow.add_node("decide_search", decide_search)
workflow.add_node("web_search", web_search)
workflow.add_node("draft_post", draft_post)

workflow.set_entry_point("decide_search")
workflow.add_edge("decide_search", "web_search")
workflow.add_edge("web_search", "draft_post")
workflow.add_edge("draft_post", END)

app = workflow.compile()

test_bot = "Bot B"
initial_state = {
    "bot_id": test_bot,
    "persona": Persona[test_bot],
    "search_query": "",
    "search_results": "",
    "final_post": {}
}

result = app.invoke(initial_state)
print("Final Post Output:")
print(json.dumps(result["final_post"], indent=2))

#