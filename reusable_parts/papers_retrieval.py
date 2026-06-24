import requests
import os
from tools import get_model
from dotenv import load_dotenv

class getReferencePaper():
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


    def query_search(self, end_date, query):
        # url="https://api.semanticscholar.org/graph/v1/paper/search/"
        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        
        query_params = {
            "query": query,
            "fields": "title,citationCount,publicationTypes,publicationDate",
            "year": f"-{end_date}",
            "limit": 1000,
            "sort": "publicationDate:asc"
        }
        print(self.api_key)
        headers = {"x-api-key": self.api_key}
        
        response = requests.get(url, params=query_params, headers=headers).json()
        
        return response

    def PaperDetails(self, paper_id, fields="title,year,abstract,authors,citationCount,venue,citations,references,tldr"):
        
        url = "https://api.semanticscholar.org/graph/v1/paper/"
        
        paper_data_query_params = {"fields": fields}
        headers = {"x-api-key": self.api_key}
        response = requests.get(
            url = url + paper_id, params=paper_data_query_params, headers=headers
        )
        
        return response.json()
    
    @staticmethod
    def prepare_papers_for_llm(list_of_papers):
        unique_papers = {}

        for query_string, query_data in list_of_papers.items():
            for paper in query_data.get('data', []):
                paper_id = paper.get('paperId')
                if paper_id and paper_id not in unique_papers:  # Skips if paper_id is None
                    paper_str = f"""Paper ID: {paper_id}
                                    Title: {paper.get('title')}
                                    Abstract: {paper.get('abstract')}
                                """
                    unique_papers[paper_id] = paper_str
                    
        paper_list = list(unique_papers.values())
                
        papers_for_llm = "\n\n---\n\n".join(paper_list)
        return papers_for_llm
    
    @staticmethod
    def prepare_papers_for_evaluation(list_of_papers):
        # Just different formatting for evaluation purpose
        unique_papers = {}

        for paper in list_of_papers:
            paper_id = paper.get('paperId')
            if paper_id and paper_id not in unique_papers:  # Skips if paper_id is None
                paper_str = f"""Paper ID: {paper_id}
                                Title: {paper.get('title')}
                                Abstract: {paper.get('abstract')}
                            """
                unique_papers[paper_id] = paper_str
                    
        paper_list = list(unique_papers.values())
                
        papers_for_eval= "\n\n---\n\n".join(paper_list)
        return papers_for_eval

 
# query = 'Computing Machinery and Intelligence'

# search_paper = getReferencePaper()
# search_paper_response = search_paper.query_search(query)


