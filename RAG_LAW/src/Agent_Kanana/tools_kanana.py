from langchain_community.tools import tool
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from src.Agent_Kanana.schemas import (UserInput, InputDocument, QueryAnswerable,
                     DocumentIssue, IssuesList, CombinedQuery, QueryList, 
                     RAGOutput, RAGList, EnoughContext, RerankItem, RerankList,
                     WebSearchQueries, WebSearchOutput, WebSearchList, 
                     ContextOutput, ContextList, AnswerOutput, AnswerEnough, Metadata)
from src.Agent_Kanana.functions import load_prompt_kanana, document_ocr, truncate_context_texts

import chromadb
from src.RAG.search_kanana_main import NaiveSearchWithAnswer
from src.Agent_Kanana.kanana_pipeline import call_kanana, call_kanana_structured

from tavily import TavilyClient
import os

@tool
def extend_query(original_query: str) -> str:
    """쿼리를 받아서 법률 용어로 확장, 재구성하는 함수"""
    try:
        system_prompt = load_prompt_kanana("extend_query_prompt")
        
        print(f"🔄 쿼리 확장 중... (원본: '{original_query}')")
        
        response = call_kanana(
            system_prompt = system_prompt,
            user_input = {"original_query": original_query},
            max_new_tokens = 256
        )
        
        # 응답 검증 및 정리 
        if not response:
            print("⚠️ Kanana가 None을 반환했습니다. 원본 쿼리를 사용합니다.")
            return original_query

        response = response.strip()
        print(response)

        if response == "":
            print("⚠️ Kanana가 빈 응답을 반환했습니다. 원본 쿼리를 사용합니다.")
            return original_query
        
        if len(response) < len(original_query) * 0.5:
            print(f"⚠️ 응답이 너무 짧습니다 ({len(response)}자). 원본 쿼리를 사용합니다.")
            return original_query

        # 최소 수정 방침: 원본보다 20단어(약 60자) 이상 길어지면 과도한 재작성으로 판단
        if len(response) > len(original_query) + 60:
            print(f"⚠️ 응답이 너무 깁니다 ({len(response)}자). 원본 쿼리를 사용합니다.")
            return original_query
        
        print(f"✅ 쿼리 확장 완료: '{original_query}' -> '{response}'")
        return response
        
    except Exception as e:
        import traceback
        print(f"❌ 쿼리 확장 중 오류 발생: {e}")
        print(f"   오류 상세: {traceback.format_exc()}")
        print(f"   원본 쿼리를 그대로 사용합니다: {original_query}")
        return original_query

@tool
def parse_document_ocr(ocr_result: Any) -> InputDocument:
    """OCR 결과를 파싱하는 함수"""
    system_prompt = load_prompt_kanana("parse_document_ocr_prompt")
    response = call_kanana(
        system_prompt = system_prompt,
        user_input = {"document_ocr": ocr_result},
        max_new_tokens = 1024
    )
    cleaned = response.strip() if response else str(ocr_result)
    return InputDocument(document=cleaned)

@tool
def check_query_answerable(extended_query: str) -> QueryAnswerable:
    """문서 없이 질문만으로 답변 가능한지 LLM에게 물어보는 함수"""
    system_prompt = load_prompt_kanana("check_query_answerable_prompt")
    response = call_kanana(
        system_prompt = system_prompt,
        user_input = {"extended_query": extended_query},
        max_new_tokens = 150
    )
    text = response.strip() if response else ""
    upper = text.upper()
    # NOT_ANSWERABLE 먼저 체크 (ANSWERABLE의 부분 문자열이므로)
    if "NOT_ANSWERABLE" in upper:
        answerable = "NOT_ANSWERABLE"
    elif "ANSWERABLE" in upper:
        answerable = "ANSWERABLE"
    else:
        answerable = "NOT_ANSWERABLE"  # 보수적 기본값
    result = QueryAnswerable(answerable=answerable, reason=text)
    print(f"📋 질문 답변 가능 여부: {result.answerable}")
    print(f"이유: {result.reason}")
    return result

@tool
def extract_issues(extended_query: str, parsed_document: InputDocument) -> IssuesList:
    """파싱된 문서를 받아와서 쟁점을 추출하는 함수 (질문 들어오면 질문 있는 거랑 관련 짓도록 설정)"""
    system_prompt = load_prompt_kanana("extract_issues_prompt")
    response = call_kanana_structured(
        system_prompt = system_prompt,
        user_input = {
            "extended_query": extended_query, 
            "parsed_document": parsed_document.document},
        output_schema = IssuesList,
        max_new_tokens = 512
    )
    return response

@tool
def search_rag(combined_queries: QueryList, rag_method: str = "naive") -> RAGList:
    """RAG 검색 기능을 수행하는 함수 (질문과 문서 쟁점들을 통합하여 검색)"""
    try:
        lawdb_path = "data/Database/LawDB"
        client = chromadb.PersistentClient(path = lawdb_path)
        collection = client.get_or_create_collection("laws")
        
        search_queries = [combined_query.to_rag_query for combined_query in combined_queries.combined_queries]
        unique_results = {} 
        
        # Kanana pipeline 가져오기
        from src.Agent_Kanana.kanana_pipeline import get_kanana_pipeline
        pipeline, _ = get_kanana_pipeline()
        
        for query in search_queries:
            naive_search_with_answer = NaiveSearchWithAnswer(collection, query, pipeline)
            formatted_docs = naive_search_with_answer.search()
            
            for doc in formatted_docs:
                text = doc.get("text", "")
                score = doc.get("relevance_score", 0.0)
                if text not in unique_results or score > unique_results[text]["relevance_score"]:
                    unique_results[text] = doc
        
        sorted_docs = sorted(unique_results.values(), key = lambda x: x.get("relevance_score", 0.0), reverse = True)
        
        rag_results = []
        for rank, doc in enumerate(sorted_docs):
            text = doc.get("text", "")
            metadata_dict = doc.get("metadata", {})
            score = doc.get("relevance_score", 0.0)
            source = metadata_dict.get("law_path", "")
            
            metadata = Metadata(
                law_path = metadata_dict.get("law_path"),
                published_date = None,
                query_used = None,
                domain = None
            )
            output = RAGOutput(
                search_rank = rank,
                text = text,
                source = source,
                metadata = metadata,
                relevance_score = score
            )
            rag_results.append(output)
        return RAGList(list_rag_results = rag_results)

    except Exception as e:
        print(f"Error in search_rag: {e}")
        return RAGList(list_rag_results = [])

@tool
def check_enough_context(combined_queries: QueryList, contexts: ContextList) -> EnoughContext:
    """RAG 결과 또는 RAG + 웹 검색 결과로 충분한 답변이 가능한지 여부를 확인하는 함수"""
    system_prompt = load_prompt_kanana("enough_context_prompt")

    queries_str = ""
    for i, query in enumerate(combined_queries.combined_queries, 1):
        queries_str += f"{i}. [{query.type}] "
        if query.type == "Question":
            queries_str += f"질문: {query.query}\n"
        else:
            queries_str += f"문서 쟁점: {query.query} (위치: {query.position}, 검토 사유: {query.reason})\n"
    
    # ContextList를 문자열로 변환 (RAG 결과와 웹 검색 결과 모두 포함 가능)
    contexts_str = ""
    for i, context in enumerate(contexts.list_contexts, 1):
        doc_type_str = "내부 DB" if context.doc_type == "Internal_DB" else "외부 웹"
        rank_str = f"순위: {context.rank}" if context.rank is not None else "순위: 미정"
        contexts_str += f"[{i}]. ({doc_type_str}) {rank_str}\n"
        contexts_str += f"내용: {context.text}\n\n"

    response = call_kanana(
        system_prompt = system_prompt,
        user_input = {
            "combined_queries": queries_str,
            "contexts": contexts_str},
        max_new_tokens = 200
    )
    text = response.strip() if response else ""
    upper = text.upper()
    # NOT_ENOUGH 먼저 체크 (ENOUGH의 부분 문자열이므로)
    if "NOT_ENOUGH" in upper:
        enough = "NOT_ENOUGH"
    elif "ENOUGH" in upper:
        enough = "ENOUGH"
    else:
        enough = "NOT_ENOUGH"  # 보수적 기본값
    return EnoughContext(enough_context=enough, reason=text)

@tool
def generate_search_queries(combined_queries: QueryList, enough_context: EnoughContext, previous_queries: Optional[WebSearchQueries] = None) -> WebSearchQueries:
    """쿼리와 피드백을 기반으로 외부 웹 검색을 위한 쿼리를 생성하는 함수"""
    try:
        queries_str = ""
        for i, query in enumerate(combined_queries.combined_queries, 1):
            queries_str += f"{i}. [{query.type}] "
            if query.type == "Question":
                queries_str += f"질문: {query.query}\n"
            else:
                queries_str += f"문서 쟁점: {query.query} (위치: {query.position}, 검토 사유: {query.reason})\n"
        
        # 이전 쿼리가 있으면 수정 프롬프트 사용, 없으면 생성 프롬프트 사용
        if previous_queries and previous_queries.web_search_queries:
            system_prompt = load_prompt_kanana("revise_search_queries_prompt")
    

            previous_queries_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(previous_queries.web_search_queries)])
            
            response = call_kanana_structured(
                system_prompt = system_prompt,
                user_input = {
                    "combined_queries": queries_str, 
                    "feedback": enough_context.reason, 
                    "previous_queries": previous_queries_str},
                output_schema = WebSearchQueries,
                max_new_tokens = 256
            )
        else:
            system_prompt = load_prompt_kanana("generate_search_queries_prompt")
    
            response = call_kanana_structured(
                system_prompt = system_prompt,
                user_input = {
                    "combined_queries": queries_str, 
                    "feedback": enough_context.reason},
                output_schema = WebSearchQueries,
                max_new_tokens = 256
            )
        
        # 쿼리 길이 검증 및 수정 (400자 제한)
        validated_queries = []
        for query in response.web_search_queries:
            if len(query) > 400:
                if " site:" in query:
                    # site: 부분을 찾아서 보존
                    parts = query.rsplit(" site:", 1)
                    if len(parts) == 2:
                        main_query = parts[0][:400 - len(parts[1]) - 7] 
                        validated_query = f"{main_query} site:{parts[1]}"
                    else:
                        validated_query = query[:400]
                else:
                    validated_query = query[:400]
                print(f"⚠️ 쿼리 길이 초과로 자동 수정: {len(query)}자 -> {len(validated_query)}자")
                validated_queries.append(validated_query)
            else:
                validated_queries.append(query)
        
        return WebSearchQueries(web_search_queries = validated_queries)
    except Exception as e:
        print(f"Error in generate_search_queries: {e}")
        fallback_queries = [q.to_rag_query for q in combined_queries.combined_queries]
        return WebSearchQueries(web_search_queries = fallback_queries if fallback_queries else ["법률 정보 검색"])

@tool
def search_web(web_search_queries: WebSearchQueries) -> WebSearchList:
    """외부 웹 검색을 수행하는 함수"""
    tavily = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))
    try:
        web_search_results = []
        for query in web_search_queries.web_search_queries:
            # 쿼리 길이 재검증 (안전장치)
            if len(query) > 400:
                print(f"⚠️ 쿼리 길이 초과 감지: {len(query)}자. 자동으로 잘라냅니다.")
                query = query[:400]
            
            # Tavily API 반환 구조: {"results": [...], "query": "...", ...}
            search_response = tavily.search(query = query, search_depth = "advanced", max_results = 3)
            results = search_response.get("results", []) if isinstance(search_response, dict) else search_response
            for result in results:
                url = result.get("url", "")
                try:
                    domain = url.split("/")[2] if len(url.split("/")) > 2 else url
                except:
                    domain = url
                
                # dict를 Metadata 객체로 변환
                metadata = Metadata(
                    law_path = None,
                    published_date = result.get("published_date", ""),
                    query_used = query,
                    domain = domain
                )
                web_search_results.append(WebSearchOutput(
                    title = result.get("title", ""),
                    text = result.get("content", ""),
                    source = url,
                    metadata = metadata,
                    relevance_score = result.get("score", 0.0)
                ))
        return WebSearchList(list_web_results = web_search_results)

    except Exception as e:
        print(f"Error in search_web: {e}")
        return WebSearchList(list_web_results = [])

@tool
def rerank_contexts(combined_queries: QueryList, all_contexts: ContextList) -> ContextList:
    """컨텍스트를 재정렬하는 함수(rank 부분에 값 채워넣기)"""
    try:
        system_prompt = load_prompt_kanana("rerank_contexts_prompt")

        # combined_queries를 문자열로 변환
        queries_str = ""
        for i, query in enumerate(combined_queries.combined_queries, 1):
            queries_str += f"{i}. [{query.type}] "
            if query.type == "Question":
                queries_str += f"질문: {query.query}\n"
            else:
                queries_str += f"문서 쟁점: {query.query} (위치: {query.position}, 검토 사유: {query.reason})\n"

        # 원본 컨텍스트를 source → ContextOutput 맵으로 보존
        original_map = {ctx.source: ctx for ctx in all_contexts.list_contexts}

        # LLM 출력 토큰 초과 방지: RAG/Web 각각 할당량을 두어 균형 있게 선택
        MAX_RAG = 8
        MAX_WEB = 7
        rag_contexts = sorted(
            [ctx for ctx in all_contexts.list_contexts if ctx.doc_type == "Internal_DB"],
            key=lambda x: x.relevance_score, reverse=True
        )[:MAX_RAG]
        web_contexts = sorted(
            [ctx for ctx in all_contexts.list_contexts if ctx.doc_type == "External_Web"],
            key=lambda x: x.relevance_score, reverse=True
        )[:MAX_WEB]
        contexts_to_llm = rag_contexts + web_contexts
        total_input = len(all_contexts.list_contexts)
        skipped_count = total_input - len(contexts_to_llm)
        if skipped_count > 0:
            print(f"⚠️ 컨텍스트 수 초과: 전체 {total_input}개 중 RAG {len(rag_contexts)}개 + 웹 {len(web_contexts)}개만 리랭킹 ({skipped_count}개 제외)")

        # LLM에는 텍스트 없이 source + 유형 + 내용만 전달 (점수 편향 방지)
        context_index_str = ""
        for i, ctx in enumerate(contexts_to_llm, 1):
            doc_type_label = "내부 DB" if ctx.doc_type == "Internal_DB" else "외부 웹"
            snippet = ctx.text[:200].replace("\n", " ")
            context_index_str += (
                f"[{i}] source: {ctx.source} | 유형: {doc_type_label} "
                f"| 내용(앞 200자): {snippet}\n"
            )

        response = call_kanana_structured(
            system_prompt = system_prompt,
            user_input = {
                "combined_queries": queries_str,
                "rag_results": context_index_str,
                "web_results": ""   # 이미 context_index_str에 통합
            },
            output_schema = RerankList,
            max_new_tokens = 1024
        )

        # LLM 결과(RerankList)를 원본 텍스트와 합쳐 ContextList로 복원
        reranked = []
        for item in response.list_contexts:
            original = original_map.get(item.source)
            if original is None:
                continue
            reranked.append(ContextOutput(
                rank            = item.rank,
                doc_type        = original.doc_type,
                text            = original.text,
                metadata        = original.metadata,
                source          = original.source,
                relevance_score = item.relevance_score,
            ))

        # LLM이 누락한 컨텍스트는 낮은 점수로 뒤에 붙임
        ranked_sources = {item.source for item in response.list_contexts}
        for ctx in all_contexts.list_contexts:
            if ctx.source not in ranked_sources:
                reranked.append(ContextOutput(
                    rank            = len(reranked) + 1,
                    doc_type        = ctx.doc_type,
                    text            = ctx.text,
                    metadata        = ctx.metadata,
                    source          = ctx.source,
                    relevance_score = 0.1,
                ))

        # Rerank 결과 점수 확인
        rag_scores = [ctx.relevance_score for ctx in reranked if ctx.doc_type == "Internal_DB"]
        web_scores = [ctx.relevance_score for ctx in reranked if ctx.doc_type == "External_Web"]
        if rag_scores:
            print(f"🔍 Rerank 후 RAG 점수: min={min(rag_scores):.2f}, max={max(rag_scores):.2f}, avg={sum(rag_scores)/len(rag_scores):.2f}")
        if web_scores:
            print(f"🔍 Rerank 후 웹 점수: min={min(web_scores):.2f}, max={max(web_scores):.2f}, avg={sum(web_scores)/len(web_scores):.2f}")

        return ContextList(list_contexts=reranked)

    except Exception as e:
        print(f"Error in rerank_contexts: {e}")
        return ContextList(list_contexts = [])

@tool
def generate_answer(extended_query: str, contexts: ContextList, extracted_issues: Optional[IssuesList], input_type: Literal["Query_Only", "Hybrid", "Error"]) -> AnswerOutput:
    """최종 컨텍스트를 받아서 응답을 생성하는 함수"""
    # 토큰 제한을 고려하여 컨텍스트를 제한
    original_count = len(contexts.list_contexts)
    max_contexts = 12  # 토큰 제한을 고려한 최대 컨텍스트 개수
    
    if original_count > max_contexts:
        print(f"⚠️ 컨텍스트가 {original_count}개로 많아 상위 {max_contexts}개만 사용합니다.")
        sorted_contexts = sorted(
            contexts.list_contexts,
            key=lambda x: (x.relevance_score, -x.rank if x.rank is not None else 0),
            reverse=True
        )
        contexts = ContextList(list_contexts=sorted_contexts[:max_contexts])
    
    # 각 컨텍스트의 텍스트 길이 제한 (원본 텍스트를 2000자로 제한)
    contexts = truncate_context_texts(contexts, max_text_length = 2000)
    
    # 프롬프트와 컨텍스트가 많을 수 있으므로 답변 토큰을 약간 줄임
    system_prompt = load_prompt_kanana("generate_answer_prompt")
    context_count = len(contexts.list_contexts)
    min_required = max(4, int(context_count * 0.7))  # 최소 70% 이상 활용
    
    response = call_kanana_structured(
        system_prompt = system_prompt,
        user_input = {
            "extended_query": extended_query, 
            "contexts": str(contexts), 
            "extracted_issues": str(extracted_issues) if extracted_issues else "없음 (Query_Only 모드)",
            "input_type": input_type,
            "context_count": str(context_count),
            "min_required_docs": str(min_required)},
        output_schema = AnswerOutput,
        max_new_tokens = 1024
    )
    
    # 참고자료 검증...
    if "## 참고자료" not in response.answer:
        print("⚠️ 경고: 참고자료 섹션이 없습니다. 자동으로 추가합니다.")
        references = "\n\n## 참고자료\n\n"
        for idx, src in enumerate(response.source, 1):
            references += f"[{idx}] {src}\n"
        response.answer = response.answer.rstrip() + references
    
    print(f"📊 문서 사용: {len(response.source)}/{context_count}개 (최소 요구: {min_required}개)")
    
    return response

@tool
def confirm_answer(extended_query: str, contexts: ContextList, extracted_issues: Optional[IssuesList], answer: AnswerOutput) -> AnswerEnough:
    """최종 답변을 받아서 적합성을 확인하는 함수"""
    # 컨텍스트 텍스트 길이 제한
    contexts = truncate_context_texts(contexts, max_text_length=2000)
    
    context_count = len(contexts.list_contexts)
    min_required = max(4, int(context_count * 0.7))
    used_count = len(answer.source)
    
    # 1차 검증: 참고자료 섹션 체크
    if "## 참고자료" not in answer.answer:
        print(f"⚠️ 답변 검증 실패: 참고자료 섹션이 없습니다.")
        return AnswerEnough(
            kind = "NOT_ENOUGH",
            feedback = f"답변에 '## 참고자료' 섹션이 누락되었습니다. 반드시 답변 마지막에 참고자료 섹션을 추가하고, "
                      f"본문에서 인용한 모든 [1], [2], [3] ... 번호에 대한 출처를 나열하세요."
        )
    
    # 2차 검증: 문서 사용 개수 체크
    if used_count < min_required:
        print(f"⚠️ 답변 검증 실패: 문서 사용 개수 부족 ({used_count}/{min_required})")
        return AnswerEnough(
            kind = "NOT_ENOUGH",
            feedback = f"제공된 {context_count}개 문서 중 {used_count}개만 사용했습니다. 최소 {min_required}개 이상의 문서를 활용하여 답변을 작성하세요. "
                      f"각 문서에서 구체적인 정보(조항, 수치, 절차, 기준 등)를 추출하여 답변에 포함하세요. "
                      f"질문과 직접 관련이 없어 보이는 문서라도 간접적으로 도움이 되는 정보(배경 지식, 관련 법령 등)가 있다면 활용하세요."
        )
    
    # 3차 검증: LLM을 통한 답변 품질 평가
    system_prompt = load_prompt_kanana("confirm_answer_prompt")
    response_text = call_kanana(
        system_prompt = system_prompt,
        user_input = {
            "extended_query": extended_query,
            "contexts": str(contexts),
            "extracted_issues": str(extracted_issues) if extracted_issues else "없음 (Query_Only 모드)",
            "answer": str(answer),
            "context_count": str(context_count),
            "min_required_docs": str(min_required)
        },
        max_new_tokens = 200
    )
    text = response_text.strip() if response_text else ""
    upper = text.upper()
    # NOT_ENOUGH 먼저 체크 (ENOUGH의 부분 문자열이므로)
    if "NOT_ENOUGH" in upper:
        kind = "NOT_ENOUGH"
    elif "ENOUGH" in upper:
        kind = "ENOUGH"
    else:
        kind = "NOT_ENOUGH"  # 보수적 기본값
    response = AnswerEnough(kind=kind, feedback=text)

    # LLM 평가 결과에 문서 사용 정보 추가
    if response.kind == "ENOUGH":
        print(f"✅ 답변 검증 통과: {used_count}/{context_count}개 문서 사용")
    else:
        print(f"⚠️ 답변 검증 실패: LLM 평가 - {response.feedback}")

    return response

@tool
def retry_answer(extended_query: str, contexts: ContextList, extracted_issues: Optional[IssuesList], previous_answer: AnswerOutput, feedback: AnswerEnough) -> AnswerOutput:
    """최종 답변과 피드백을 받아서 재생성하는 함수"""
    # 컨텍스트 텍스트 길이 제한
    contexts = truncate_context_texts(contexts, max_text_length = 2000)
    
    system_prompt = load_prompt_kanana("retry_answer_prompt")
    response = call_kanana_structured(
        system_prompt = system_prompt,
        user_input = {
            "extended_query": extended_query, 
            "contexts": str(contexts), 
            "extracted_issues": str(extracted_issues) if extracted_issues else "없음 (Query_Only 모드)",
            "previous_answer": str(previous_answer), 
            "feedback": str(feedback)},
        output_schema = AnswerOutput,
        max_new_tokens = 1024
    )
    return response