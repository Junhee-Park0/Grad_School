"""
Agent 전체 실행 파일 - Kanana 버전
"""

from langgraph.graph import StateGraph, END, START

import os
from dotenv import load_dotenv
load_dotenv(".env")

# Config 및 Logger import
from utils.config import Config
from utils.logger import logger, log_agent_action, setup_logger

from src.Agent_Kanana.tools_kanana import (extend_query, parse_document_ocr, check_query_answerable, extract_issues, 
                    search_rag, check_enough_context, generate_search_queries, search_web, 
                    rerank_contexts, generate_answer, confirm_answer, retry_answer)
from src.Agent_Kanana.schemas import (UserInput, InputDocument, DocumentIssue, IssuesList, 
                    CombinedQuery, QueryList, RAGOutput, RAGList, 
                    EnoughContext, WebSearchQueries, WebSearchOutput, WebSearchList, 
                    ContextOutput, ContextList, AnswerOutput, AnswerEnough)
from src.Agent_Kanana.states import LegalAgentState
from src.Agent_Kanana.functions import (load_prompt_kanana, route_by_input_type, document_ocr, filter_low_relevance_contexts,
                    route_by_enough_context, route_by_enough_answer, should_regenerate, route_after_document_parsing)
# Kanana 버전의 nodes를 import
from src.Agent_Kanana.nodes_kanana_track import (routing_node, query_rewriting_node, 
                            document_parsing_node, issue_extracting_node, 
                            rag_searching_node, context_evaluating_node, web_searching_node, 
                            context_reranking_node, context_filtering_node, answer_generating_node, answer_evaluating_node, answer_regenerating_node)
from src.Agent_Kanana.kanana_pipeline import get_kanana_pipeline

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
    """Legal Agent 메인 함수 - Kanana 버전"""
    # ============================================================================
    # 환경 설정 - 사용자 입력으로 설정
    # ============================================================================
    print("=" * 60)
    print("⚖️  Legal Agent - Kanana 버전")
    print("=" * 60)
    print("🔧 환경 설정:")
    
    global logger 
    enable_local_logging = input("로컬 로깅을 활성화하시겠습니까? (Y/N): ").strip().lower()
    if enable_local_logging == 'y':
        Config.ENABLE_LOCAL_LOGGING = True
    else:
        Config.ENABLE_LOCAL_LOGGING = False
    
    # 로컬 로깅 설정에 따라 logger 재설정
    logger = setup_logger()
    
    print("\n📋 현재 설정:")
    config_summary = Config.get_config_summary()
    for key, value in config_summary.items():
        print(f" • {key}: {value}")
    print()

    if Config.ENABLE_LOCAL_LOGGING:
        now_str = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/law_agent_{now_str}.log"
        print(f"📝 로컬 로그: {log_filename}")
        logger.info(f"로컬 로깅 활성화: {now_str}")
    else:
        print("📝 로컬 로그: 비활성화")
        log_filename = None

    # ============================================================================
    # Kanana 파이프라인 초기화
    # ============================================================================
    print("\n" + "=" * 60)
    print("🔄 Kanana 모델 초기화 중...")
    print("⚠️  처음 실행 시 모델 다운로드 및 로드에 몇 분이 걸릴 수 있습니다.")
    print("   (이후 실행 시에는 이미 로드된 모델을 사용하므로 빠릅니다.)\n")
    
    if Config.ENABLE_LOCAL_LOGGING:
        log_agent_action("Kanana 모델 로드 시작")
    
    get_kanana_pipeline()
    
    print("\n" + "=" * 60)
    if Config.ENABLE_LOCAL_LOGGING:
        log_agent_action("Kanana 모델 로드 완료")
    
    # ============================================================================
    # Agent 초기화 (모델은 이미 로드됨)
    # ============================================================================
    print("🔧 Agent 워크플로우 초기화 중...")
    agent = legal_agent()
    print("✅ Agent 준비 완료!")
    
    # ============================================================================
    # 대화형 모드: 여러 질문을 연속으로 처리
    # ============================================================================
    print("=" * 60)
    print("💬 대화형 모드")
    print("=" * 60)
    print("💡 팁: 모델이 이미 로드되어 있어 빠르게 답변할 수 있습니다!")
    print("   종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.\n")
    
    question_count = 0
    while True:
        try:
            question_count += 1
            print(f"\n{'─' * 60}")
            print(f"질문 #{question_count}")
            print(f"{'─' * 60}")
            
            query = input("\n질문을 입력해주세요 (종료: quit/exit/q): ").strip()
            
            # 종료 조건
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 프로그램을 종료합니다.")
                break
            
            document_path = input("\n함께 첨부하실 문서가 있다면 경로를 입력해주세요 (없으면 Enter): \n").strip()
            
            if Config.ENABLE_LOCAL_LOGGING:
                log_agent_action("사용자 입력 수신", {
                    "question_number": question_count,
                    "has_query": bool(query),
                    "has_document": bool(document_path)
                })
            
            # ============================================================================
            # Legal Agent 실행
            # ============================================================================
            original_input = UserInput(
                query = query if query else None, 
                document_path = document_path if document_path else None
            )
            
            if Config.ENABLE_LOCAL_LOGGING:
                log_agent_action("에이전트 워크플로우 시작")
            
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
                continue  # 다음 질문으로 계속
            
            if result.get("answer"):
                answer = result.get("answer").answer
                if Config.ENABLE_LOCAL_LOGGING:
                    logger.info("\n" + "=" * 30)
                    logger.info("[최종 답변]")
                    logger.info(f"\n{answer}")
                    logger.info("\n" + "=" * 30)
                print("\n" + "=" * 60)
                print("🔎 최종 답변")
                print("=" * 60)
                print(f"\n{answer}\n")
                
                if Config.ENABLE_LOCAL_LOGGING:
                    from utils.logger import log_conversation
                    log_conversation(query, answer)

            print("=" * 60)
            print("✅ 답변 완료!")
            print("=" * 60)
            
            if Config.ENABLE_LOCAL_LOGGING:
                log_agent_action("에이전트 워크플로우 완료")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자가 중단했습니다.")
            break
        except Exception as e:
            print(f"\n❌  Legal Agent 실행 중 오류가 발생했습니다: {e}")
            print("=" * 60)
            if Config.ENABLE_LOCAL_LOGGING:
                from utils.logger import log_error
                log_error(e, "legal_agent_main")
            print("\n다음 질문을 계속 진행할 수 있습니다.\n")
            continue
    
    print("\n" + "=" * 60)
    print(f"🎉 총 {question_count - 1}개의 질문을 처리했습니다!")
    print("=" * 60) 

if __name__ == "__main__":
    legal_agent_main()