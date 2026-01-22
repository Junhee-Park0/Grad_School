"""
생성된 VectorDB를 활용해서 검색 기능을 구현
- Naive RAG 기반 검색 기능
- Hybrid RAG 기반 검색 기능
- OpenAI 활용하여 답변 생성
"""
import chromadb
import re, os, math, pickle, sys 
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import openai
from dotenv import load_dotenv
load_dotenv()

from src.RAG.naive_search import NaiveSearchEngine
# search, save_filtered, load_filtered 함수
from src.RAG.embedding import LawEmbeddings
# create_query_embedding 함수

# 문서 필터링 진행
class NaiveSearchWithAnswer():
    def __init__(self, collection, query : str, model : str = "gpt-4o-mini"):
        self.collection = collection
        self.query = query
        self.model = model
        self.client = openai.OpenAI()
        self.query_embedding = LawEmbeddings().create_query_embedding(query)
        self.search_engine = NaiveSearchEngine(collection, self.query_embedding, top_k = 15, save_path = "Database/FilteredDB")

    def search(self, where : Optional[Dict] = None):
        search_results = self.search_engine.search(self.query_embedding, where = where)
        return search_results
    
    def format_filtered_docs(self, filtered_docs : List[Dict]) -> str:
        if not filtered_docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i , doc in enumerate(filtered_docs):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            law_name = metadata.get("law_name", "")
            law_path = metadata.get("law_path", "")
            score = doc.get("relevance_score", 0.0)

            context = f"""
            --- Document {i+1} (Source : {law_name}, Path : {law_path}, Score : {score:.3f}) ---
            Text : {text}
            """
            formatted_docs.append(context)
        
        return "\n".join(formatted_docs)

    def generate_answer(self, filtered_docs : List[Dict]):
        formatted_docs = self.format_filtered_docs(filtered_docs)

        prompt = f"""
        You are a helpful assistant that can answer the question based on the following documents:
        Question: {self.query}
        Filtered Documents: {formatted_docs}
        Information about the filtered documents:
        - Text : the text of the document. you can use it to generate the overall content for the answer.
        - Source : the source of the document (the law_name of the text).
        - Path : the path of the document 
        - Score : the relevance score of the document to the question, the higher the score, the more relevant the document is to the question
        Cautions when generating the answer:
        - You should generate the answer **only** based on the filtered documents, and the relevance score.
        - You should mention the source and path of the document at the end of the answer. 
          : (출처 : the source1 - the path1, 출처 : the source2 - the path2, ...)
        """
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = [{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content, formatted_docs

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."

    def filter_and_generate_answer(self, top_k : int = 10, where : Optional[Dict] = None):
        filtered_docs = self.search(where = where)
        answer, formatted_docs = self.generate_answer(filtered_docs)
        return answer, formatted_docs

if __name__ == "__main__":
    # if len(sys.argv) < 2: 
    #     print("검색할 쿼리를 입력해주세요.")
    #     sys.exit(1)
    # query = sys.argv[1]
    query = input("검색할 쿼리를 입력해주세요: ")
    print("--------------------------------")
    print(f"Query : {query}")
    print("--------------------------------")
    # ChromaDB 경로 설정
    lawdb_path = "data/Database/LawDB"
    client = chromadb.PersistentClient(path = lawdb_path)
    collection = client.get_or_create_collection("laws")
    # NaiveSearchWithAnswer 객체 생성
    naive_search_with_answer = NaiveSearchWithAnswer(collection, query)
    naive_search_with_answer.search()
    # 검색 결과 생성
    # answer, formatted_docs = naive_search_with_answer.filter_and_generate_answer()  
    # print(f"Answer : {answer}")
    # print("--------------------------------")
    # print(f"Formatted Documents : \n\n {formatted_docs}")
    # print("--------------------------------")
    # # 필터링된 결과 저장
    # naive_search_with_answer.search_engine.save_filtered(query)
    

