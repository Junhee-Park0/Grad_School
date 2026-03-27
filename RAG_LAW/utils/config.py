"""
Legal Agent 설정 파일

환경변수나 직접 설정으로 Agent의 동작을 제어합니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Agent 전역 설정"""
    
    # ============================================================================
    # 로깅 설정 (기본값)
    # ============================================================================
    ENABLE_LOCAL_LOGGING = False

    # ============================================================================
    # LangSmith 설정 (OpenAI 버전에서 사용)
    # ============================================================================
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "Legal_Agent")

    # ============================================================================
    # 모델 설정
    # ============================================================================
    # Kanana 모델
    KANANA_MODEL_NAME = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
    KANANA_MAX_NEW_TOKENS = int(os.getenv("KANANA_MAX_NEW_TOKENS", "512"))
    
    # OpenAI 모델 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ============================================================================
    # 경로 설정
    # ============================================================================
    LOG_DIR = "logs"
    KANANA_MODEL_PATH = "E:\Kanana_Model"
    
    @classmethod
    def get_config_summary(cls):
        """현재 설정 요약"""
        return {
            "로컬 로깅": "활성화" if cls.ENABLE_LOCAL_LOGGING else "비활성화",
            "모델": cls.KANANA_MODEL_NAME,
        }
