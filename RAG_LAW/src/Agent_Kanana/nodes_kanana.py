"""
Legal Agent 구성 노드들 - Kanana 버전
"""

from langgraph.graph import StateGraph, END, START

import os
from dotenv import load_dotenv
load_dotenv(".env")

# Logger import
from utils.logger import logger, log_agent_action

# Kanana 버전의 tools를 import
from src.Agent_Kanana.tools_kanana import (extend_query, parse_document_ocr, check_query_answerable, extract_issues,
                    search_rag, check_enough_context, generate_search_queries, search_web, 
                    rerank_contexts, generate_answer, confirm_answer, retry_answer)
from src.Agent_Kanana.schemas import (UserInput, InputDocument, DocumentIssue, IssuesList, 
                    CombinedQuery, QueryList, RAGOutput, RAGList, 
                    EnoughContext, WebSearchQueries, WebSearchOutput, WebSearchList, 
                    ContextOutput, ContextList, AnswerOutput, AnswerEnough)
from src.Agent_Kanana.states import LegalAgentState
from src.Agent_Kanana.functions import load_prompt_kanana, determine_input_type, document_ocr, filter_low_relevance_contexts

# Nodes
def routing_node(state: LegalAgentState) -> LegalAgentState:
    """입력을 받아서 입력 형식을 결정하는 노드"""
    input_type = determine_input_type(state["original_input"])
    print(f"입력 형식 : {input_type}")

    logger.info(f"> 질문| {state['input_query']}")
    logger.info(f"> 입력 형식| {input_type}")
    return {"input_type": input_type, "context_retry_count": 0, "answer_retry_count": 0}

def query_rewriting_node(state: LegalAgentState) -> LegalAgentState:
    """질문을 재작성하는 노드"""
    logger.info(f"\n[쿼리 재작성 노드]")
    extended_query = extend_query.invoke({"original_query": state["input_query"]})
    print("-" * 50)
    print("✒️  쿼리 재작성")
    print("-" * 50)
    print(f"기존 쿼리 : {state['input_query']}")
    print(f"확장 쿼리 : {extended_query}")

    logger.info(f"> 재작성된 쿼리| {extended_query}")

    return {"extended_query": extended_query}

def document_parsing_node(state: LegalAgentState) -> LegalAgentState:
    """문서를 파싱하는 노드"""
    logger.info(f"\n[문서 파싱 노드]")
    try:
        document_ocr_result = document_ocr(state["original_input"].document_path)
        if not document_ocr_result or document_ocr_result.strip() == "":
            raise ValueError("문서 OCR 결과가 비어있습니다.")
        parsed_document = parse_document_ocr.invoke({"ocr_result": document_ocr_result})
        print("✅ 문서 파싱 완료")
        logger.info(f"> 문서 파싱 완료")
        return {"parsed_document": parsed_document}
        
    except Exception as e:
        print(f"❌ 문서 파싱 중 오류가 발생했습니다: {e}")
        
        if state.get("input_query") and state["input_query"].strip():
            print("📝 문서 없이 질문만으로 답변이 가능한지 확인합니다.")
            answerable = check_query_answerable.invoke({"extended_query": state["extended_query"]})
            
            if answerable.answerable == "ANSWERABLE":
                print("✅ 질문만으로 답변 가능합니다. Query_Only 모드로 전환합니다.")
                logger.info(f"> 질문만으로 답변 가능| Query_Only 모드로 전환")
                return {
                    "input_type": "Query_Only", "doc_parse_failed": True, "query_answerable": answerable}
            else:
                print("❌ 문서 없이는 답변할 수 없는 질문입니다. 워크플로우를 종료합니다.")
                logger.info(f"> 문서 없이 답변 불가| 워크플로우 종료")
                return {
                    "input_type": "Error", "error_message": f"문서 처리 실패 & 문서 없이 답변 불가: {str(e)}", "query_answerable": answerable}
        else:
            print("❌ 문서 처리 실패 및 질문이 없습니다. 워크플로우를 종료합니다.")
            logger.info(f"> 문서 처리 실패 및 질문이 없음| 워크플로우 종료")
            return {"input_type": "Error", "error_message": f"문서 처리 실패: {str(e)}"}

def issue_extracting_node(state: LegalAgentState) -> LegalAgentState:
    """문서에서 쟁점을 추출하는 노드"""
    logger.info(f"\n[문서 파싱 및 쟁점 추출 노드]")
    extracted_issues = extract_issues.invoke({
        "extended_query": state["extended_query"], 
        "parsed_document": state["parsed_document"]
    })
    print("-" * 50)
    print("📄  문서 파싱 및 쟁점 추출")
    print("-" * 50)
    print(f"추출된 쟁점 수 : {len(extracted_issues.issues)}")
    logger.info(f"> 추출된 쟁점 수| {len(extracted_issues.issues)}")
    return {"extracted_issues": extracted_issues}

def rag_searching_node(state: LegalAgentState) -> LegalAgentState:
    """RAG를 검색하는 노드"""
    logger.info(f"\n[RAG 검색 노드]")
    # 쿼리 합치기
    combined_queries = []
    combined_queries.append(CombinedQuery(
        query = state["extended_query"], 
        type = "Question"))
    
    # extracted_issues가 있으면 추가 (Hybrid 경로에서만)
    extracted_issues = state.get("extracted_issues")
    if extracted_issues:
        for issue in extracted_issues.issues:
            combined_queries.append(CombinedQuery(
                query = issue.issue, 
                type = "Document", 
                position = issue.position, 
                reason = issue.reason))
    
    combined_queries = QueryList(combined_queries = combined_queries)
    
    rag_results = search_rag.invoke({"combined_queries": combined_queries})
    initial_contexts_list = [rag_result.to_context for rag_result in rag_results.list_rag_results]
    initial_contexts = ContextList(list_contexts = initial_contexts_list)
    print("-" * 50)
    print("📑  RAG 검색")
    print("-" * 50)
    print(f"검색된 문서 수 : {len(rag_results.list_rag_results)}")
    logger.info(f"> 검색된 문서 수| {len(rag_results.list_rag_results)}")
    return {"combined_queries": combined_queries, "rag_results": rag_results, "all_contexts": initial_contexts_list, "reranked_contexts": initial_contexts} # 웹 검색이 없는 경우를 대비한 초기 컨텍스트
    
def context_evaluating_node(state: LegalAgentState) -> LegalAgentState:
    """컨텍스트 평가 노드"""
    logger.info(f"\n[컨텍스트 평가 노드]")
    current_retry = state.get("context_retry_count", 0)
    print(f"컨텍스트 재생성 횟수 : {current_retry}")
    logger.info(f"> 컨텍스트 재생성 횟수| {current_retry}")
    if current_retry >= 1:
        logger.info(f"> 컨텍스트 재생성 횟수가 최대 횟수에 도달하였습니다. 현재까지의 컨텍스트를 사용합니다.")
        print("❗컨텍스트 재생성 횟수가 최대 횟수에 도달하였습니다. 현재까지의 컨텍스트를 사용합니다.\n")
        enough_context = EnoughContext(
            enough_context = "ENOUGH", 
            reason = "Maximum number of retries reached. Proceeding with collected contexts."
        )
        logger.info(f"> 컨텍스트 평가 결과| {enough_context}")
        return {"enough_context": enough_context, "context_retry_count": current_retry}
    
    if "reranked_contexts" in state and state["reranked_contexts"]:
        contexts_to_evaluate = state["reranked_contexts"]
    else:
        contexts_to_evaluate = ContextList(
            list_contexts=[rag_result.to_context for rag_result in state["rag_results"].list_rag_results]
        )
    enough_context = check_enough_context.invoke({
        "combined_queries": state["combined_queries"], 
        "contexts": contexts_to_evaluate
    })
    print("-" * 50)
    print("🔍  컨텍스트 평가")
    print("-" * 50)
    print(f"컨텍스트 평가 결과 : {enough_context}")
    return {"enough_context": enough_context, "context_retry_count": current_retry + 1}

def web_searching_node(state: LegalAgentState) -> LegalAgentState:
    """웹 검색하는 노드"""
    logger.info(f"\n[웹 검색 노드]")
    # 이전 쿼리, 웹 검색 결과가 있으면 전달 
    previous_queries = state.get("web_search_queries")
    previous_web_results = state.get("web_search_results")
    
    web_search_queries = generate_search_queries.invoke({
        "combined_queries": state["combined_queries"], 
        "enough_context": state["enough_context"],
        "previous_queries": previous_queries
    })
    new_web_search_results = search_web.invoke({"web_search_queries": web_search_queries})
    
    # 이전 결과와 새 결과를 합침 (중복 제거)
    if previous_web_results and previous_web_results.list_web_results:
        existing_sources = {result.source for result in previous_web_results.list_web_results}
        new_results = [
            result for result in new_web_search_results.list_web_results
            if result.source not in existing_sources
        ]
        combined_results = previous_web_results.list_web_results + new_results
        web_search_results = WebSearchList(list_web_results = combined_results)
        print("-" * 50)
        print("🌐  웹 검색")
        print("-" * 50)
        print(f"이전 검색 결과: {len(previous_web_results.list_web_results)}개")
        print(f"새 검색 결과: {len(new_web_search_results.list_web_results)}개 (중복 제외: {len(new_results)}개)")
        print(f"총 검색 결과: {len(web_search_results.list_web_results)}개")
    else:
        web_search_results = new_web_search_results
        print("-" * 50)
        print("🌐  웹 검색")
        print("-" * 50)
        if previous_queries:
            print("🔄  이전 쿼리를 수정하여 재검색합니다.")
        print(f"검색된 문서 수 : {len(web_search_results.list_web_results)}")

    new_contexts = [web_result.to_context for web_result in web_search_results.list_web_results]
    
    logger.info(f"> 웹 검색 결과 개수| {len(web_search_results.list_web_results)}개")
    return {"web_search_queries": web_search_queries, "web_search_results": web_search_results, "all_contexts": new_contexts}

def context_filtering_node(state: LegalAgentState) -> LegalAgentState:
    """관련도가 낮은 컨텍스트를 필터링하는 노드"""
    logger.info(f"\n[컨텍스트 필터링 노드]")
    # 첫 실행에는 all_contexts 사용
    if not state.get("filtered_contexts"):
        if "all_contexts" not in state or not state["all_contexts"]:
            return {}
        print("-" * 50)
        print("🔍  컨텍스트 필터링 (첫 실행)")
        contexts_list = ContextList(list_contexts = state["all_contexts"])
        filtered_contexts = filter_low_relevance_contexts(contexts = contexts_list) 
        print("-" * 50)
        logger.info(f"> 컨텍스트 필터링 결과| {len(filtered_contexts.list_contexts)}개")
        return {"filtered_contexts": filtered_contexts.list_contexts}
    # 두 번째부터는 filtered_contexts와 all_contexts를 합침
    else:
        existing_filtered = state["filtered_contexts"]
        all_contexts_list = state.get("all_contexts", [])
        
        existing_rag = [context for context in existing_filtered if context.doc_type == "Internal_DB"]
        existing_web = {context.source for context in existing_filtered}
        new_web_contexts = [context for context in all_contexts_list if context.doc_type == "External_Web" and context.source not in existing_web] # 중복 제거해서 합치기
        
        if new_web_contexts:
            new_web_list = ContextList(list_contexts = new_web_contexts)
            filtered_new_web = filter_low_relevance_contexts(contexts = new_web_list)
            existing_web = [context for context in existing_filtered if context.doc_type == "External_Web"]
            combined = existing_rag + existing_web + filtered_new_web.list_contexts
        else:
            combined = existing_filtered
        
        print("-" * 50)
        print("🔍  컨텍스트 필터링 (재실행)")
        print(f"기존 필터링: {len(existing_filtered)}개")
        print(f"새 웹 contexts: {len(new_web_contexts)}개")
        print(f"최종: {len(combined)}개")
        print("-" * 50)
        logger.info(f"> 컨텍스트 필터링 결과| {len(combined)}개")
        return {"filtered_contexts": combined}
    
def context_reranking_node(state: LegalAgentState) -> LegalAgentState:
    """컨텍스트 재정렬 노드 (filtered_contexts 사용)"""
    logger.info(f"\n[컨텍스트 재정렬 노드]")
    contexts_to_rerank = state.get("filtered_contexts", state.get("all_contexts", []))
    if not contexts_to_rerank:
        print("⚠️ 재정렬할 컨텍스트가 없습니다. 빈 컨텍스트로 처리합니다.")
        logger.info(f"> 재정렬할 컨텍스트가 없습니다. 빈 컨텍스트로 처리합니다.")
        return {"reranked_contexts": ContextList(list_contexts=[])}
    
    contexts_list = ContextList(list_contexts = contexts_to_rerank)
    reranked_contexts = rerank_contexts.invoke({
        "combined_queries": state["combined_queries"],
        "all_contexts": contexts_list
    })
    
    # 원본 텍스트 복원
    original_sources = {context.source: context for context in contexts_to_rerank}
    reranked_original_contexts = []

    for reranked_context in reranked_contexts.list_contexts:
        original_context = original_sources.get(reranked_context.source)
        if original_context:
            reordered_original_context = ContextOutput(
                rank = reranked_context.rank,
                doc_type = original_context.doc_type,
                text = original_context.text,
                metadata = original_context.metadata,
                source = original_context.source,
                relevance_score = reranked_context.relevance_score
            )
            reranked_original_contexts.append(reordered_original_context)
    
    logger.info(f"> 컨텍스트 재정렬 결과| {len(reranked_original_contexts)}개")
    return {"reranked_contexts": ContextList(list_contexts=reranked_original_contexts)}

def answer_generating_node(state: LegalAgentState) -> LegalAgentState:
    """답변을 생성하는 노드"""
    logger.info(f"\n[답변 생성 노드]")
    print("-" * 50)
    print("💬  답변 생성")
    print("-" * 50)
    print(f"\n답변 생성 중...")
    if "reranked_contexts" in state and state["reranked_contexts"]:
        contexts = state["reranked_contexts"]
    else:
        contexts = ContextList(list_contexts=state.get("all_contexts", []))
    
    answer = generate_answer.invoke({
        "extended_query": state["extended_query"], 
        "extracted_issues": state.get("extracted_issues"),
        "contexts": contexts, 
        "input_type": state["input_type"],
        "max_new_tokens": 1024
    })
    print(f"생성된 답변 : {answer.answer}")
    logger.info(f"> 생성된 답변| {answer.answer}")
    return {"answer": answer, "answer_history": [answer]}
    
def answer_evaluating_node(state: LegalAgentState) -> LegalAgentState:
    """답변을 평가하는 노드"""
    logger.info(f"\n[답변 평가 노드]")
    print("-" * 50)
    print("⚖️  답변 평가")
    print("-" * 50)
    print(f"\n답변 평가 중...")
    if "reranked_contexts" in state and state["reranked_contexts"]:
        contexts = state["reranked_contexts"]
    else:
        contexts = ContextList(list_contexts=state.get("all_contexts", []))
    
    answer_enough = confirm_answer.invoke({
        "extended_query": state["extended_query"], 
        "extracted_issues": state.get("extracted_issues"),
        "contexts": contexts, 
        "answer": state["answer"]
    })
    print(f"답변 평가 결과 : {answer_enough.kind}")
    logger.info(f"> 답변 평가 결과| {answer_enough.kind}")
    return {"answer_enough": answer_enough}
    
def answer_regenerating_node(state: LegalAgentState) -> LegalAgentState:
    """답변을 재생성하는 노드"""
    logger.info(f"\n[답변 재생성 노드]")
    print("-" * 50)
    print("🔄  답변 재생성")
    print("-" * 50)
    current_retry = state.get("answer_retry_count", 0)
    logger.info(f"> 답변 재생성 횟수| {current_retry}")
    if current_retry >= 1:
        print("❗답변 재생성 횟수가 최대 횟수에 도달하였습니다. 가장 높은 신뢰도의 답변을 선택합니다.")
        logger.info(f"> 답변 재생성 횟수가 최대 횟수에 도달하였습니다. 가장 높은 신뢰도의 답변을 선택합니다.")
        answer_history = state.get("answer_history", [])
        if answer_history:
            best_answer = max(answer_history, key = lambda x: x.confidence_score)
        else:
            best_answer = state.get("answer", None)
        print(f"가장 높은 신뢰도의 답변 : {best_answer}")
        # answer_retry_count를 should_regenerate의 종료 임계값(3) 이상으로 설정하여 루프 탈출
        return {"answer": best_answer, "answer_retry_count": 3}
    
    if "reranked_contexts" in state and state["reranked_contexts"]:
        contexts = state["reranked_contexts"]
    else:
        contexts = ContextList(list_contexts=state.get("all_contexts", []))
    
    new_answer = retry_answer.invoke({
        "extended_query": state["extended_query"], 
        "extracted_issues": state.get("extracted_issues"),
        "contexts": contexts, 
        "previous_answer": state["answer"], 
        "feedback": state["answer_enough"],
        "max_new_tokens": 1024
    })
    print(f"재생성된 답변 : {new_answer.answer}")
    # answer_history는 operator.add reducer가 자동으로 이어붙이므로 새 답변만 반환
    return {"answer": new_answer, "answer_retry_count": current_retry + 1, "answer_history": [new_answer]}
