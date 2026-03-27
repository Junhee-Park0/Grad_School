import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 모듈 수준 싱글톤 — 프로세스 전체에서 한 번만 로드됨
_embedding_model: Optional[SentenceTransformer] = None
_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def _get_embedding_model() -> SentenceTransformer:
    """BGE-M3 모델 싱글톤을 반환한다. 최초 호출 시에만 로드된다."""
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📥 BGE-M3 임베딩 모델 로드 중... (device: {device})")
        try:
            _embedding_model = SentenceTransformer(
                _EMBEDDING_MODEL_NAME, device=device, local_files_only=True
            )
        except Exception:
            print("⚠️ 로컬 캐시 로드 실패 — HuggingFace에서 다운로드합니다.")
            _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME, device=device)
        _embedding_model.max_seq_length = 512
        print("✅ BGE-M3 임베딩 모델 로드 완료")
    return _embedding_model


class LawEmbeddings:
    def __init__(self, model_name: str = _EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self) -> SentenceTransformer:
        """싱글톤 모델을 반환한다."""
        return _get_embedding_model()

    def load_model(self):
        """하위 호환성 유지용 — 실제 로드는 _get_embedding_model()이 처리한다."""
        _ = self.model  # 싱글톤 초기화 트리거

    def create_embeddings(self, laws_parsed : List[Dict]) -> List[np.ndarray]:
        """임베딩 생성"""        
        self.load_model()
        texts = [doc.get('text',"") for doc in laws_parsed]
        try:
            embeddings = self.model.encode(
                texts,
                batch_size = 128,
                show_progress_bar = True,
                convert_to_numpy = True,
                normalize_embeddings = True
            )
            return embeddings
        except Exception as e:
            print(f"임베딩 실패 : {e}")
            raise

    def create_query_embedding(self, text : str) -> np.ndarray:
        """쿼리 임베딩 생성"""
        self.load_model()

        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy = True,
                normalize_embeddings = True
            )[0]
            return embedding
        except Exception as e:
            print(f"쿼리 임베딩 실패 : {e}")
            raise

    def save_embeddings(self, embeddings : List[np.ndarray], filename : str):
        """임베딩 저장"""
        np.save(filename, embeddings)

if __name__ == "__main__":
    with open("data/Laws/Processed/laws_parsed.json", "r", encoding = 'utf-8') as f:
        laws_parsed = json.load(f)
    law_emb = LawEmbeddings()
    laws_embedded = law_emb.create_embeddings(laws_parsed)
    laws_embedded = laws_embedded.astype(np.float32)
    law_emb.save_embeddings(laws_embedded, "data/Laws/Processed/laws_embedded.npy")
    print("임베딩 완료 !")