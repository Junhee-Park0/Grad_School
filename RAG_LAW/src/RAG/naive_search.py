import re
import os
import math
import pickle
import chromadb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from src.RAG.embedding import LawEmbeddings

# ChromaDB 텔레메트리 비활성화 (오류 방지)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# posthog 텔레메트리 오류 방지를 위한 패치
try:
    import posthog
    # capture() 함수를 빈 함수로 패치하여 오류 방지
    def dummy_capture(*args, **kwargs):
        pass
    posthog.capture = dummy_capture
except ImportError:
    pass

class NaiveSearchEngine():
    def __init__(self, collection, query_embedding, normalize : bool = True, top_k : int = 5, save_path : str = "Database/FilteredDB"):
        """
        vector_db의 자료를 기반으로 query와 관련 있는 문서 k개 필터링, 응답 생성
        (Naive RAG)
        - collection : 벡터 DB
        - query_embedding : 임베딩된 질의
        - normalize : 임베딩 정규화 여부
        - top_k : 검색 결과 상위 k개
        - save_path : 필터링한 유사 자료들 저장할 폴더 경로
        """
        self.collection = collection
        self.query_embedding = query_embedding
        self.normalize = normalize
        self.top_k = top_k
        self.documents : List[Dict] = []
        self.embeddings : Optional[np.ndarray] = None
        self.metadatas : Optional[List[Dict]] = None
        self.save_path = Path(save_path)

    def normalize_emb(self, x : np.ndarray) -> np.ndarray:
        """유사도 계산을 위해 임베딩 정규화"""
        # 1차원인 경우
        if x.ndim == 1:
            denom = np.linalg.norm(x) + 1e-10
            return x / denom
        # 2차원인 경우
        else:
            denom = np.linalg.norm(x, axis = 1, keepdims = True) + 1e-10
            return x / denom
    
    def search(self, query_embedding : np.ndarray, where : Optional[Dict] = None) -> List[Dict]:
        """
        기본 벡터 검색(Naive RAG)
        - where : 조건 필터링 (metadata 활용해서)
            - 예시 : filter = {"law_name" : "개인정보 보호법"}
                -> where = filter 로 넣으면 개인정보 보호법만 필터링 가능
        """
        try:
            if self.normalize:
                query_embedding = self.normalize_emb(query_embedding)

            results = self.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = self.top_k,
                where = where,
                include = ['documents', 'metadatas', 'distances', 'embeddings']
            )

            # 필터링 결과 저장을 위한 업데이트
            self.documents = results['documents'][0]
            self.metadatas = results['metadatas'][0]
            self.embeddings = np.array(results['embeddings'][0])

            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    "index" : i, # 검색 결과 순서 기반
                    "text" : results['documents'][0][i],
                    "metadata" : results['metadatas'][0][i],
                    "relevance_score" : 1 - results["distances"][0][i],
                    "search_rank" : i # 벡터 검색 순위
                }
                formatted_results.append(result)
            return formatted_results
        
        except Exception as e:
            print(f"검색 중 오류 발생 : {e}")
            self.documents = []
            self.metadatas = []
            self.embeddings = None
            return []

    def save_filtered(self, filename : str):
        """유사한 자료들을 모아 저장해둠"""
        self.save_path.mkdir(parents = True, exist_ok = True)
        # 윈도우 금지 문자: \ / : * ? " < > |
        safe_filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        save_path = self.save_path / f"{safe_filename}.pickle"

        with open(save_path, "wb") as f:
            pickle.dump({
                "documents" : self.documents,
                "metadatas" : self.metadatas,
                "embeddings" : self.embeddings.astype(np.float32),
                "normalize" : self.normalize
            }, f)
            
    def load_filtered(self, filename : str) -> pd.DataFrame:
        safe_filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        load_path = self.save_path / f"{safe_filename}.pickle"
        if not load_path.exists():
            return None
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        combined_data = []
        for doc, meta in zip(data['documents'], data['metadatas']):
            row_data = meta.copy()
            row_data["document"] = doc
            combined_data.append(row_data)
        df = pd.DataFrame(combined_data).drop(columns = ["eff_date", "hang_num", "section_type", "jomun_num"])
        return df

if __name__ == "__main__":
    # 1. ChromaDB 버전 확인
    print(f"ChromaDB 버전: {chromadb.__version__}")

    # 2. 데이터베이스 상태 확인
    lawdb_path = "data/Database/LawDB"

    client = chromadb.PersistentClient(path=str(lawdb_path))
    collection = client.get_or_create_collection("laws")

    # 3. 컬렉션 정보 확인
    print(f"문서 수: {collection.count()}")

    # 4. 첫 번째 문서의 임베딩 확인
    try:
        # 모든 문서 가져오기
        all_docs = collection.get(limit = 1, include=['embeddings'])
        if all_docs['embeddings'] is not None and len(all_docs['embeddings']) > 0:
            first_emb = np.array(all_docs['embeddings'][0])
            print(f"첫 번째 임베딩 shape: {first_emb.shape}")
            print(f"첫 번째 임베딩 norm: {np.linalg.norm(first_emb)}")
            
            # 정규화 여부 확인
            if abs(np.linalg.norm(first_emb) - 1.0) < 0.01:
                print("✅ 임베딩이 정규화되어 있음")
            else:
                print("❌ 임베딩이 정규화되지 않음")
        else:
            print("❌ 임베딩이 없음")
            
    except Exception as e:
        print(f"임베딩 확인 중 오류: {e}")