"""
Agent 전체 실행 파일 - OpenAI 버전
LangSmith 추적 항상 활성화
"""

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

import os
from dotenv import load_dotenv
load_dotenv(".env")

from utils.config import Config
from utils.logger import logger, log_agent_action

from src.Agent_OpenAI.tools import (extend_query, parse_document_ocr, check_query_answerable, extract_issues, 
                    search_rag, check_enough_context, generate_search_queries, search_web, 
                    rerank_contexts, generate_answer, confirm_answer, retry_answer)
from src.Agent_OpenAI.schemas import (UserInput, InputDocument, DocumentIssue, IssuesList, 
                    CombinedQuery, QueryList, RAGOutput, RAGList, 
                    EnoughContext, WebSearchQueries, WebSearchOutput, WebSearchList, 
                    ContextOutput, ContextList, AnswerOutput, AnswerEnough)
from src.Agent_OpenAI.states import LegalAgentState
from src.Agent_OpenAI.functions import (load_prompt, route_by_input_type, document_ocr, filter_low_relevance_contexts,
                    route_by_enough_context, route_by_enough_answer, should_regenerate, route_after_document_parsing)
from src.Agent_OpenAI.nodes import (routing_node, query_rewriting_node, 
                            document_parsing_node, issue_extracting_node, 
                            rag_searching_node, context_evaluating_node, web_searching_node, 
                            context_reranking_node, context_filtering_node, answer_generating_node, answer_evaluating_node, answer_regenerating_node)

def legal_agent():
    """Legal Agent 그래프"""
    workflow = StateGraph(LegalAgentState)

    # 노드 추가
    workflow.add_node("input_router", routing_node)
    workflow.add_node("query_rewriter", query_rewriting_node)
    workflow.add_node("document_parser", document_parsing_node)
    workflow.add_node("issue_extractor", issue_extracting_node)
    workflow.add_node("rag_searcher", rag_searching_node)
    workflow.add_node("context_evaluator", context_evaluating_node)
    workflow.add_node("web_searcher", web_searching_node)
    workflow.add_node("context_reranker", context_reranking_node)
    workflow.add_node("context_filter", context_filtering_node)
    workflow.add_node("answer_generator", answer_generating_node)
    workflow.add_node("answer_evaluator", answer_evaluating_node)
    workflow.add_node("answer_regenerator", answer_regenerating_node)

    # 엣지 추가
    workflow.add_edge(START, "input_router")
    workflow.add_edge("input_router", "query_rewriter")
    workflow.add_conditional_edges(
        "query_rewriter", 
        route_by_input_type,
        {
            "Query_Only" : "rag_searcher", 
            "Hybrid" : "document_parser",
            "Error" : END
        }
    )
    workflow.add_conditional_edges(
        "document_parser",
        route_after_document_parsing,
        {
            "Hybrid" : "issue_extractor",
            "Query_Only" : "rag_searcher",
            "Error" : END
        }
    )
    workflow.add_edge("issue_extractor", "rag_searcher")
    workflow.add_edge("rag_searcher", "context_evaluator")
    workflow.add_conditional_edges(
        "context_evaluator",
        route_by_enough_context,
        {
            "ENOUGH" : "answer_generator",
            "NOT_ENOUGH" : "web_searcher"
        }
    )
    workflow.add_edge("web_searcher", "context_filter")
    workflow.add_edge("context_filter", "context_reranker")
    workflow.add_edge("context_reranker", "context_evaluator")
    workflow.add_edge("answer_generator", "answer_evaluator")
    workflow.add_conditional_edges(
        "answer_evaluator",
        route_by_enough_answer,
        {
            "ENOUGH" : END,
            "NOT_ENOUGH" : "answer_regenerator"
        }
    )
    workflow.add_conditional_edges(
        "answer_regenerator",
        should_regenerate,
        {
            "YES" : "answer_evaluator",
            "NO" : END
        }
    )

    return workflow.compile()

def legal_agent_main():
    """Legal Agent 메인 함수 - OpenAI 버전"""
    
    # ============================================================================
    # OpenAI 버전 설정 (LangSmith 항상 활성화)
    # ============================================================================
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if Config.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = Config.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = Config.LANGSMITH_PROJECT
    # ============================================================================
    # 사전 설정 정보 출력
    # ============================================================================
    print("=" * 60)
    print("⚖️  Legal Agent - OpenAI 버전")
    print("=" * 60)
    print("\n📋 설정:")
    print(f"  • 로컬 로깅: {'활성화' if Config.ENABLE_LOCAL_LOGGING else '비활성화'}")
    print(f"  • LangSmith 추적: 항상 활성화 ✅")
    if Config.LANGSMITH_API_KEY:
        print(f"  • LangSmith 프로젝트: {Config.LANGSMITH_PROJECT}")
    else:
        print("  ⚠️  LANGSMITH_API_KEY가 설정되지 않았습니다!")
    print()

    if Config.ENABLE_LOCAL_LOGGING:
        print(f"📝 로컬 로그: logs/law_agent_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.log")
    else:
        print("📝 로컬 로그: 비활성화 (콘솔만)")

    print(f"🔗 LangSmith: {Config.LANGSMITH_PROJECT}")
    print("=" * 60)
    
    query = input("\n질문을 입력해주세요: ").strip()
    document_path = input("\n함께 첨부하실 문서가 있다면 경로를 입력해주세요: \n").strip()
    
    if Config.ENABLE_LOCAL_LOGGING:
        log_agent_action("사용자 입력 수신", {
            "has_query": bool(query),
            "has_document": bool(document_path)
        })
    # ============================================================================
    # Legal Agent 실행 시작
    # ============================================================================
    try:
        agent = legal_agent()
        original_input = UserInput(
            query = query if query else None, 
            document_path = document_path if document_path else None
        )
        
        if Config.ENABLE_LOCAL_LOGGING:
            log_agent_action("에이전트 워크플로우 시작 (OpenAI)")
        
        result = agent.invoke({
            "original_input" : original_input,
            "input_query" : query if query else ""
        })
        
        if result.get("input_type") == "Error":
            error_message = result.get("error_message", "알 수 없는 오류가 발생했습니다.")
            print("=" * 60)
            print(f"❌ 오류가 발생했습니다: {error_message}")
            print("=" * 60)
            if Config.ENABLE_LOCAL_LOGGING:
                logger.error(f"워크플로우 오류: {error_message}")
            return
        
        if result.get("answer"):
            answer = result.get("answer").answer
            print("=" * 60)
            print("\n\n")
            print(f"🔎 최종 답변 : {answer}")
            
            if Config.ENABLE_LOCAL_LOGGING:
                from utils.logger import log_conversation
                log_conversation(query, answer)

        print("=" * 60)
        print("🎉 Legal Agent 실행 완료 !")
        print("=" * 60)
        
        if Config.ENABLE_LOCAL_LOGGING:
            log_agent_action("에이전트 워크플로우 완료 (OpenAI)")
            
    except Exception as e:
        print(f"❌  Legal Agent 실행 중 오류가 발생했습니다: {e}")
        print("=" * 60)
        if Config.ENABLE_LOCAL_LOGGING:
            from utils.logger import log_error
            log_error(e, "legal_agent_main (OpenAI)")

if __name__ == "__main__":
    legal_agent_main()