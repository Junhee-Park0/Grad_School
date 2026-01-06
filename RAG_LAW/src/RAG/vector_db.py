import os

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

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import gc
import psutil

class LawVectorDB:
    def __init__(self, vectordb_path : str = "data/Database/LawDB", vectordb_name : str = "laws"):
        self.vectordb_path = vectordb_path
        if not os.path.exists(self.vectordb_path):
            os.makedirs(self.vectordb_path)
        self.vectordb_name = vectordb_name
        print(f"=== VectorDB 경로: {self.vectordb_path} ===")

        """
        client - ChromaDB 클라이언트
        collection - ChromaDB 컬렉션 (문서 + 임베딩 + 메타데이터가 실제로 저장되는 공간)
        documents - 원본 문서 리스트 (List[Dict])
        """
        self.client = None
        self.collection = None
        self.search_engine = None
        self.documents = []

    def initialize_db(self):
        # client, collection, search_engine 초기화된 상태라면 불러오기
        if self.client is None:
            print("ChromaDB 클라이언트 초기화 중...")
            # telemetry 문제 방지를 위한 부분 
            settings = Settings(
                anonymized_telemetry = False,
                allow_reset = True
            )
            self.client = chromadb.PersistentClient(
                path=str(self.vectordb_path),
                settings=settings
            )
            print("ChromaDB 클라이언트 초기화 성공")
            
            # 가장 간단한 컬렉션 생성 (HNSW 없이)
            print("기본 컬렉션 생성 중...")
            try:
                # 기존 컬렉션 시도
                self.collection = self.client.get_collection(self.vectordb_name)
                print(f"기존 컬렉션 '{self.vectordb_name}' 로드 완료")
            except:
                # 컬렉션이 없으면 새로 생성 (HNSW 없이)
                # metadata 파라미터를 제거하거나 None으로 설정 (빈 딕셔너리는 허용되지 않음)
                self.collection = self.client.create_collection(
                    name=self.vectordb_name
                )
                print(f"새 컬렉션 '{self.vectordb_name}' 생성 완료")

            # 문서 수 확인
            doc_count = self.collection.count()
            print(f"총 {doc_count}개의 문서 발견")

    def get_memory_usage(self):
        """현재 메모리 사용량 반환 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def add_documents(self, documents : List[Dict], embeddings : List[np.ndarray]):
        """ChromaDB에 문서 추가"""
        self.initialize_db()
        self.documents = documents

        try:
            # 배치 처리 - 커널 그만 죽어라
            batch_size = 1000
            total_laws = len(documents)
            # 처음 메모리 사용량 확인
            print(f"=== 시작 메모리: {self.get_memory_usage():.1f}MB ===")

            for i in range(0, total_laws, batch_size):
                end_idx = min(i + batch_size, total_laws)
                batch_laws = documents[i : end_idx]
                batch_embeddings = embeddings[i : end_idx]

                # 배치 길이 검증
                if len(batch_laws) != len(batch_embeddings):
                    raise ValueError(f"문서 수({len(batch_laws)})와 임베딩 수({len(batch_embeddings)})가 불일치합니다. i={i}, end_idx={end_idx}")

                if (i // batch_size + 1) % 10 == 0:
                    print(f"배치 {i // batch_size + 1}/{(total_laws + batch_size - 1)//batch_size} : {end_idx - i}개 문서 저장 중... (메모리: {self.get_memory_usage():.1f}MB)")

                ids = [f"doc_{j}" for j in range(i, end_idx)]
                texts = [doc.get("embedding_text", "") for doc in batch_laws]

                metadatas = []
                for doc in batch_laws:
                    metadata = {
                        "law_name" : str(doc.get("law_meta", {}).get("law_name", "")),
                        "eff_date" : str(doc.get("law_meta", {}).get("eff_date", "")),
                        "law_path" : str(doc.get("path", "")),
                        "section_type" : str(doc.get("section_type", "")),
                        "junmun_num" : str(doc.get("junmun_num", "")),
                        "jomun_num" : str(doc.get("jomun_num", "")),
                        "hang_num" : str(doc.get("hang_no", ""))
                    }
                    metadatas.append(metadata)

                self.collection.add(
                    ids = ids,
                    documents = texts,
                    embeddings = [emb.tolist() for emb in batch_embeddings],
                    metadatas = metadatas
                )

                # print(f"문서 처리 : {end_idx} / {total_laws} 완료 (메모리: {self.get_memory_usage():.1f}MB)")

                # 메모리 압력 완화
                del batch_laws
                del batch_embeddings
                del ids
                del texts
                del metadatas
                # del embeddings_list
                
                # 가비지 컬렉션 강제 실행
                gc.collect()
                
                # 메모리 사용량이 너무 높으면 경고
                if self.get_memory_usage() > 4000:  # 4GB 이상이면 경고
                    print(f"[주의] 메모리 사용량이 높습니다: {self.get_memory_usage():.1f}MB")

        except Exception as e:
            print(f"문서 추가 중 오류 발생 : {e}")
            print(f"오류 발생 시 메모리: {self.get_memory_usage():.1f}MB")
            raise

    def search(self, query_embedding: List[float], n_results: int = 5):
        """벡터 검색 수행"""
        if self.collection is None:
            self.initialize_db()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 결과 정리
            search_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    search_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i] if results['documents'] else "",
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'relevance_score': 1.0 - results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            return search_results
            
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            raise

    def get_collection_info(self):
        """컬렉션 정보 조회"""
        if self.collection is None:
            self.initialize_db()
        
        try:
            count = self.collection.count()
            return {
                'name': self.vectordb_name,
                'count': count,
                'path': str(self.vectordb_path)
            }
        except Exception as e:
            print(f"컬렉션 정보 조회 중 오류 발생: {e}")
            return None