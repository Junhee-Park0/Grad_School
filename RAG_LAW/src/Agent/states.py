from typing import List, Dict, Literal, TypedDict, Optional, Annotated
import operator
from src.Agent.schemas import (UserInput, InputDocument, QueryAnswerable, 
                     DocumentIssue, IssuesList, CombinedQuery, QueryList, RAGOutput, RAGList, 
                     EnoughContext, WebSearchQueries, WebSearchOutput, WebSearchList, 
                     ContextOutput, ContextList, AnswerOutput, AnswerEnough)

class LegalAgentState(TypedDict, total = False):
    """Agent State (Query, Document Shared)"""
    # 공통
    original_input : UserInput # 사용자 입력 (원본)
    input_type : Literal["Query_Only", "Hybrid", "Error"]
    input_query : str # 사용자 질문 (원본)
    extended_query : str # 사용자 질문 (확장 후)
    query_answerable : QueryAnswerable # 질문 답변 가능 여부
    error_message : str # 오류 메시지 (input_type == "Error"일 때 사용)
    doc_parse_failed : bool # 문서 파싱 실패 여부 (Query_Only로 전환된 경우 True)

    # [RAG]
    # rag_method : Literal["naive", "hybrid"]

    # [Document]
    parsed_document : InputDocument # 파싱된 문서
    extracted_issues : IssuesList # 추출된 쟁점들
    risk_summary : str # 리스크 요약본

    # [Combined Query]
    combined_queries : QueryList # 질문 + 문서 쟁점들 통합한 쿼리

    # 공통
    rag_results : RAGList # 검색된 문서들
    web_search_queries : WebSearchQueries # 웹 검색 쿼리
    web_search_results : WebSearchList # 웹 검색 결과
    all_contexts : Annotated[List[ContextOutput], operator.add] # 내부 + 외부 검색 결과 (RAG + Web만 추가, 필터링/재정렬은 덮어쓰기)
    filtered_contexts : List[ContextOutput] # 필터링된 컨텍스트 (덮어쓰기용)
    reranked_contexts : ContextList # 재정렬된 컨텍스트 (덮어쓰기용)
    enough_context : EnoughContext # 외부 검색 필요성 확인
    context_retry_count : int # 컨텍스트 재생성 횟수 (2회 이상 실패 시 종료)

    # [Answer]
    answer : AnswerOutput # 최종 답변
    answer_enough : AnswerEnough # 최종 답변 적합성 확인
    answer_retry_count : int # 답변 재생성 횟수 (3회 이상 실패 시 종료)
    answer_history : Annotated[List[AnswerOutput], operator.add] # 답변 재생성 기록

    