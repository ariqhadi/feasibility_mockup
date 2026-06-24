from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool 
import os, json

from dotenv import load_dotenv
load_dotenv()

###############################
## CONFIGURATION FUNCTIONS
###############################

def get_model():
    # Load LLM model based on configuration file
    
    config = get_config()
    model_name = config.get("model")
    temperature = config.get("temperature", 0)
    
    if "gemini" in model_name:
        return ChatGoogleGenerativeAI(model=model_name,
                                      google_api_key=os.getenv("GOOGLE_API_KEY"))
    elif "gpt" in model_name:
        return ChatOpenAI(
            temperature=temperature,  # Set the temperature for the model's responses
            model_name=model_name,  # Specify the model name
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def get_config():
    # Read configuration file
    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f) 
    return data

def call_llm(prompt):
    llm = get_model()
    response = llm.invoke(prompt)
    return response.content

###############################
## AGENTIC TOOLS
###############################

# Tools that work with pre-retrieved papers
def analyze_papers_tool(focus_area: str, papers: str) -> str:
    """Analyze retrieved papers focusing on a specific area"""
    return f"Analysis of papers focusing on '{focus_area}': Found relevant insights from the pre-retrieved literature."

def extract_paper_details_tool(paper_criteria: str, papers: str) -> str:
    """Extract specific details from papers based on criteria"""
    # Parse papers and extract relevant details
    lines = papers.split('\n\n---\n\n')
    relevant_papers = []
    
    for paper in lines[:3]:  # Limit to first 3 papers for demonstration
        if paper.strip():
            relevant_papers.append(f"Paper analysis for '{paper_criteria}': {paper[:200]}...")
    
    return f"Extracted details based on '{paper_criteria}': {len(relevant_papers)} papers analyzed."

def compare_methodologies_tool(methodology_aspect: str, papers: str) -> str:
    """Compare methodologies in retrieved papers"""
    return f"Methodology comparison for '{methodology_aspect}': Analyzed methodological approaches in retrieved papers."

def execute_tool(tool_name: str, params: dict, retrieved_papers: str) -> str:
    """Execute the specified tool with parameters and pre-retrieved papers"""
    if tool_name == "analyze_papers":
        return analyze_papers_tool(params.get("focus_area", ""), retrieved_papers)
    elif tool_name == "extract_details":
        return extract_paper_details_tool(params.get("criteria", ""), retrieved_papers)
    elif tool_name == "compare_methods":
        return compare_methodologies_tool(params.get("aspect", ""), retrieved_papers)
    else:
        return f"Tool {tool_name} executed with params {params}"
    
    # Define tools
tools = [
    Tool(
        name="analyze_papers",
        func=analyze_papers_tool,
        description="Analyze retrieved papers focusing on specific aspects"
    ),
    Tool(
        name="extract_details", 
        func=extract_paper_details_tool,
        description="Extract specific methodological or technical details"
    ),
    Tool(
        name="compare_methods",
        func=compare_methodologies_tool,
        description="Compare methodologies across retrieved papers"
    ),
]


