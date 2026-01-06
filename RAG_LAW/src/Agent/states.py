from typing import List, Dict, Literal, TypedDict, Optional
from schemas import (
    InputType, InputQuery, InputDocument, DocumentIssue, 
    RAGOutput, EnoughContext, WebSearchOutput, ContextOutput, 
    AnswerOutput, AnswerEnough
    )

class LegalAgentState(TypedDict, total = False):
    """Agent State (Query, Document Shared)"""
    # 공통
    input_type : InputType # "Question", "Document"
    query : InputQuery # 사용자 질문 (문서와 함께 덧붙인 질문 포함)
    query_summary : str # 질문 요약 (FilteredDB 저장용)

    # [RAG]
    # rag_method : Literal["naive", "hybrid"]

    # [Document]
    document_ocr : str # OCR 결과
    extracted_issues : List[DocumentIssue] # 추출된 쟁점들
    risk_summary : str # 리스크 요약본

    # 공통
    retreived_documents : List[RAGOutput] # 검색된 문서들
    web_search_results : List[WebSearchOutput] # 웹 검색 결과
    final_context : List[ContextOutput] # 내부 + 외부 검색 결과 최종 컨텍스트
    enough_context : EnoughContext # 외부 검색 필요성 확인

    # 최종 답변
    final_answer : AnswerOutput # 최종 답변
    answer_correct : AnswerEnough # 최종 답변 적합성 확인
    retry_count : int # 답변 재생성 횟수 (3회 이상 실패 시 종료)


    