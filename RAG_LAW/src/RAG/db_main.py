import sys, os
import numpy as np 
import pandas as pd
import json
import shutil
from pathlib import Path
import time
import gc
import psutil

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
from src.RAG.vector_db import LawVectorDB

def get_memory_usage():
    """현재 메모리 사용량 반환 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 벡터 DB 생성
def create_vectordb():
    """
    임베딩 결과 불러오고 
    VectorDB 생성하기
    """
    # 0. 기존 VectorDB 완전 삭제
    vectordb_path = "data/Database/LawDB"
    
    if vectordb_path.exists():
        print("기존 VectorDB 삭제 중...")
        shutil.rmtree(vectordb_path)
        print("✅ 기존 VectorDB 삭제 완료")

    law_vectordb = None  # 초기화
    try:
        print(f"=== 시작 메모리 사용량: {get_memory_usage():.1f}MB ===")
        
        # 1. 문서들 불러오기
        print("=== 문서들 로드 중 ===")
        doc_start_time = time.time()
        documents_path = "data/Laws/Processed/laws_parsed.json"
        
        with open(documents_path, "r", encoding = "utf-8") as f:
            documents = json.load(f)
        
        doc_loading_time = time.time() - doc_start_time
        print(f"=== 문서들 로드 완료!({doc_loading_time:.2f}초) ===")
        print(f"=== 문서 로드 후 메모리: {get_memory_usage():.1f}MB ===")
        
        # 2. 임베딩 결과 불러오기
        print("=== 임베딩 결과 로드 중 ===")
        emb_start_time = time.time()
        embeddings_path = "data/Laws/Processed/laws_embedded.npy"
        embeddings = np.load(embeddings_path)
        
        emb_loading_time = time.time() - emb_start_time
        print(f"=== 임베딩 결과 로드 완료!({emb_loading_time:.2f}초) ===")
        print(f"=== 임베딩 로드 후 메모리: {get_memory_usage():.1f}MB ===")
        
        # 3. 벡터 DB 삭제 후 생성
        print("=== Vector DB 업데이트 중 ===")
        vectordb_start_time = time.time()
        vector_db_path = "data/Database/LawDB"
        
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
            print("=== 기존 VectorDB 삭제 완료! ===")
        
        # 메모리 정리
        gc.collect()
        print(f"=== 가비지 컬렉션 후 메모리: {get_memory_usage():.1f}MB ===")
        
        # 새로운 벡터 DB 생성
        law_vectordb = LawVectorDB(vectordb_path = str(vector_db_path), vectordb_name = "laws")
        law_vectordb.initialize_db()
        
        # 컬렉션 정보 확인
        info = law_vectordb.get_collection_info()
        print(f"컬렉션 정보: {info}")
        
        # 전체 문서 처리 (메모리 절약을 위해 vector_db.py에서 배치 처리)
        print(f"=== 전체 {len(documents)}개 문서 처리 시작 ===")
        
        law_vectordb.add_documents(documents, embeddings)
        
        vector_db_time = time.time() - vectordb_start_time
        print(f"=== Vector DB 생성 완료!({vector_db_time:.2f}초) ===")
        print(f"=== 최종 메모리 사용량: {get_memory_usage():.1f}MB ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print(f"오류 발생 시 메모리: {get_memory_usage():.1f}MB")
        raise

    finally:
        # 메모리 정리
        gc.collect()
        print("=== 메모리 정리 완료 ===")
        print("=== 최종 결과 ===")
        # law_vectordb가 생성되었는지 확인
        if law_vectordb is not None:
            info = law_vectordb.get_collection_info()
            if info is not None:
                print(f"- 총 문서 수 : {info['count']}")
                print(f"- 컬렉션 경로 : {info['path']}")
                print(f"- 컬렉션 이름 : {info['name']}")
            else:
                print("- 컬렉션 정보를 가져올 수 없습니다.")
        else:
            print("- VectorDB가 생성되지 않았습니다.")


if __name__ == "__main__":
    create_vectordb()