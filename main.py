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

persons={
    "Bot A":"I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "Bot B":"I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "Bot C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI.",
}
print("\n-------------Phase 1: Vector-Based Persona Matching (The Router)-------------\n")
vector_store = Chroma(embedding_function=embeddings)

docs=[Document(page_content=text, metadata={"bot_id":name}) for name, text in persons.items()]
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
