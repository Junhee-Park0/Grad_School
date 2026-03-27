"""
생성된 VectorDB를 활용해서 검색 기능을 구현
- Naive RAG 기반 검색 기능
- Hybrid RAG 기반 검색 기능
- Kanana 활용하여 답변 생성
"""
import os

# ChromaDB 텔레메트리 비활성화 (오류 방지)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# posthog 텔레메트리 오류 방지를 위한 패치
try:
    import posthog
    def dummy_capture(*args, **kwargs):
        pass
    posthog.capture = dummy_capture
except ImportError:
    pass

import chromadb
import re, math, pickle, sys 
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import torch    
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.RAG.naive_search import NaiveSearchEngine
# search, save_filtered, load_filtered 함수
from src.RAG.embedding import LawEmbeddings
# create_query_embedding 함수

# 문서 필터링 진행
class NaiveSearchWithAnswer():
    def __init__(self, collection, query : str, pipeline : pipeline):
        self.collection = collection
        self.query = query
        self.pipeline = pipeline
        self.query_embedding = LawEmbeddings().create_query_embedding(query)
        self.search_engine = NaiveSearchEngine(collection, self.query_embedding, top_k = 10, save_path = "Database/FilteredDB")

    def search(self, where : Optional[Dict] = None):
        return self.search_engine.search(self.query_embedding, where = where)
    
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

        full_prompt = f"""
        당신은 주어진 문서에 기반하여 질문에 답변하는 유용한 법률 어시스턴트입니다. 

        [검색된 문서]
        {formatted_docs}

        [문서 정보]
        - Text : 답변 생성에 사용해야 할 문서의 내용입니다.
        - Source : 문서의 출처 (법률 이름)입니다.
        - Path : 문서의 법률 경로입니다.
        - Score : 질문과 문서의 관련성 점수입니다. (높을수록 관련성이 높음)

        [답변 생성 시 주의사항]
        - 답변은 **반드시** 위에 제공된 [검색된 문서] 내용과 관련성 점수(Score)에만 기반해야 합니다.
        - 답변 마지막에는 근거가 된 문서의 출처와 경로를 다음과 같은 형식으로 명시해야 합니다. (예: (출처: 법률이름1 - 경로1, 출처 : 법률이름2 - 경로2, ...))

        [질문]
        {self.query}
        """

        messages = [
            {"role": "system", "content": full_prompt}
        ]
        try:
            response = self.pipeline(
                messages,
                max_new_tokens = 512,
                temperature = None, # 0.0 과 같은 역할
                do_sample = False, # temperature = 0.0 과 같은 역할
                return_full_text = False,
                eos_token_id = self.pipeline.tokenizer.eos_token_id # 답변 생성 중 토크나이저의 '문장 끝' 토큰을 생성하면 멈추도록 설정
                )
            answer = response[0]["generated_text"]
            return answer, formatted_docs

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "답변을 생성하는 중 에러가 발생했습니다.", formatted_docs

    def filter_and_generate_answer(self, top_k : int = 10, where : Optional[Dict] = None):
        # 요청마다 검색 상위 개수를 동적으로 변경
        self.search_engine.top_k = top_k
        filtered_docs = self.search(where = where)
        answer, formatted_docs = self.generate_answer(filtered_docs)
        return answer, formatted_docs

if __name__ == "__main__":
    
    query = input("검색할 쿼리를 입력해주세요: ")
    # ChromaDB 경로 설정
    lawdb_path = "data/Database/LawDB"
    client = chromadb.PersistentClient(path = lawdb_path)
    collection = client.get_or_create_collection("laws")
    # pipeline 설정
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from utils.config import Config
    model = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
    model_path = Config.KANANA_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex = True)
    # CUDA 미사용 환경용: bitsandbytes/4bit 양자화 없이 CPU로 모델 로딩
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "cpu",
        torch_dtype = torch.float32
    )
    # pipeline 생성
    pipeline = pipeline("text-generation", model = model, tokenizer = tokenizer)
    # NaiveSearchWithAnswer 객체 생성
    naive_search_with_answer = NaiveSearchWithAnswer(collection, query, pipeline)
    # 검색 결과 생성
    answer, formatted_docs = naive_search_with_answer.filter_and_generate_answer()  
    print("--------------------------------")
    print(f"Query : {query}")
    print("--------------------------------")
    print(f"Answer : {answer}")
    print("--------------------------------")
    print(f"Formatted Documents : \n\n {formatted_docs}")
    print("--------------------------------")
    # 필터링된 결과 저장
    naive_search_with_answer.search_engine.save_filtered(query)

