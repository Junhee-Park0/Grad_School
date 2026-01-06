import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class LawEmbeddings:
    def __init__(self, model_name : str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """모델 로딩"""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, device = self.device)
                self.model.max_seq_length = 512
            except Exception as e:
                print(f"모델 로드 실패 : {e}")
                raise 

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