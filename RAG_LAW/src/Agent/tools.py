from langchain_community.tools import tool
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from src.Agent.schemas import (UserInput, InputDocument, QueryAnswerable,
                     DocumentIssue, IssuesList, CombinedQuery, QueryList, 
                     RAGOutput, RAGList, EnoughContext, WebSearchQueries, WebSearchOutput, WebSearchList, 
                     ContextOutput, ContextList, AnswerOutput, AnswerEnough, Metadata)
from src.Agent.functions import load_prompt, document_ocr, truncate_context_texts
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

import chromadb
from src.RAG.search_openai_main import NaiveSearchWithAnswer

from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

@tool
def extend_query(original_query: str) -> str:
    """쿼리를 받아서 법률 용어로 확장, 재구성하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("extend_query_prompt")
    chain = prompt | llm
    response = chain.invoke({"original_query": original_query})
    return response.content

@tool
def parse_document_ocr(ocr_result: Any) -> InputDocument:
    """OCR 결과를 파싱하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("parse_document_ocr_prompt")
    structured_llm = llm.with_structured_output(InputDocument)
    chain = prompt | structured_llm
    response = chain.invoke({"document_ocr": ocr_result})
    return response

@tool
def check_query_answerable(extended_query: str) -> QueryAnswerable:
    """문서 없이 질문만으로 답변 가능한지 LLM에게 물어보는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("check_query_answerable_prompt")
    structured_llm = llm.with_structured_output(QueryAnswerable)
    chain = prompt | structured_llm
    response = chain.invoke({"extended_query": extended_query})
    
    print(f"📋 질문 답변 가능 여부: {response.answerable}")
    print(f"이유: {response.reason}")
    
    return response

@tool
def extract_issues(extended_query: str, parsed_document: InputDocument) -> IssuesList:
    """파싱된 문서를 받아와서 쟁점을 추출하는 함수 (질문 들어오면 질문 있는 거랑 관련 있도록 설정)"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("extract_issues_prompt")
    structured_llm = llm.with_structured_output(IssuesList)
    chain = prompt | structured_llm
    response = chain.invoke({
        "query": extended_query, 
        "document": parsed_document.document})
    return response

@tool
def search_rag(combined_queries: QueryList, rag_method: str = "naive") -> RAGList:
    """RAG 검색 기능을 수행하는 함수 (질문과 문서 쟁점들을 통합하여 검색)"""
    try:
        lawdb_path = "data/Database/LawDB"
        client = chromadb.PersistentClient(path = lawdb_path)
        collection = client.get_or_create_collection("laws")
        
        search_queries = [combined_query.to_rag_query for combined_query in combined_queries.combined_queries]
        unique_results = {} # 중복 방지용
        
        for query in search_queries:
            naive_search_with_answer = NaiveSearchWithAnswer(collection, query)
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
            # dict를 Metadata 객체로 변환
            metadata = Metadata(
                law_path=metadata_dict.get("law_path"),
                published_date=None,
                query_used=None,
                domain=None
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
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("enough_context_prompt")
    structured_llm = llm.with_structured_output(EnoughContext)
    chain = prompt | structured_llm

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
        contexts_str += f"내용: {context.text}\n"
        contexts_str += f"출처: {context.source}\n"
        contexts_str += f"관련성 점수: {context.relevance_score}\n\n"

    response = chain.invoke({
        "combined_queries": queries_str, 
        "reranked_contexts": contexts_str}) 
    return response

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
            llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
            prompt = load_prompt("revise_search_queries_prompt")
            structured_llm = llm.with_structured_output(WebSearchQueries)
            chain = prompt | structured_llm
            
            previous_queries_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(previous_queries.web_search_queries)])
            
            response = chain.invoke({
                "combined_queries": queries_str,
                "feedback": enough_context.reason,
                "previous_queries": previous_queries_str
            })
        else:
            llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
            prompt = load_prompt("generate_search_queries_prompt")
            structured_llm = llm.with_structured_output(WebSearchQueries)
            chain = prompt | structured_llm
            response = chain.invoke({
                "combined_queries": queries_str,
                "feedback": enough_context.reason
            })
        
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
                print(f"원본: {query[:100]}...")
                print(f"수정: {validated_query[:100]}...")
                validated_queries.append(validated_query)
            else:
                validated_queries.append(query)
        
        return WebSearchQueries(web_search_queries = validated_queries)
    except Exception as e:
        print(f"Error in generate_search_queries: {e}")
        return WebSearchQueries(web_search_queries = [])

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
                    law_path=None,
                    published_date=result.get("published_date", ""),
                    query_used=query,
                    domain=domain
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
        llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
        prompt = load_prompt("rerank_contexts_prompt")
        structured_llm = llm.with_structured_output(ContextList)
        chain = prompt | structured_llm
        
        # combined_queries를 문자열로 변환
        queries_str = ""
        for i, query in enumerate(combined_queries.combined_queries, 1):
            queries_str += f"{i}. [{query.type}] "
            if query.type == "Question":
                queries_str += f"질문: {query.query}\n"
            else:
                queries_str += f"문서 쟁점: {query.query} (위치: {query.position}, 검토 사유: {query.reason})\n"
        
        rag_results_org = [context for context in all_contexts.list_contexts if context.doc_type == "Internal_DB"]
        web_results_to_context_org = [context for context in all_contexts.list_contexts if context.doc_type == "External_Web"]
        
        if web_results_to_context_org:
            rag_results_to_context = []
            for context in rag_results_org:
                reset_context = ContextOutput(
                    rank = context.rank,
                    doc_type = context.doc_type,
                    text = context.text,
                    metadata = context.metadata,
                    source = context.source,
                    relevance_score = 0.0 
                )
                rag_results_to_context.append(reset_context)
        
            truncated_contexts = []
            for context in web_results_to_context_org:
                if len(context.text) > 1000:
                    truncated_context = ContextOutput(
                        rank = context.rank,
                        doc_type = context.doc_type,
                        text = context.text[:1000] + "...(텍스트가 길어 일부만 표시)",
                        metadata = context.metadata,
                        source = context.source,
                        relevance_score = context.relevance_score
                    )
                    truncated_contexts.append(truncated_context)
                else:
                    truncated_contexts.append(context)
            web_results_to_context = truncated_contexts
        
        else:
            rag_results_to_context = rag_results_org
            web_results_to_context = []

        response = chain.invoke({
            "combined_queries": queries_str, 
            "rag_results": rag_results_to_context, 
            "web_results": web_results_to_context})
        
        # Rerank 결과 점수 확인
        rag_scores = [ctx.relevance_score for ctx in response.list_contexts if ctx.doc_type == "Internal_DB"]
        web_scores = [ctx.relevance_score for ctx in response.list_contexts if ctx.doc_type == "External_Web"]
        if rag_scores:
            print(f"🔍 Rerank 후 RAG 점수: min={min(rag_scores):.2f}, max={max(rag_scores):.2f}, avg={sum(rag_scores)/len(rag_scores):.2f}")
        if web_scores:
            print(f"🔍 Rerank 후 웹 점수: min={min(web_scores):.2f}, max={max(web_scores):.2f}, avg={sum(web_scores)/len(web_scores):.2f}")
        
        return response

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
    
    # gpt-4o 사용: 토큰 제한을 고려하여 max_tokens 조정
    # 프롬프트와 컨텍스트가 많을 수 있으므로 답변 토큰을 약간 줄임
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("generate_answer_prompt")
    structured_llm = llm.with_structured_output(AnswerOutput)
    chain = prompt | structured_llm
    context_count = len(contexts.list_contexts)
    min_required = max(4, int(context_count * 0.7))  # 최소 70% 이상 활용
    
    response = chain.invoke({
        "extended_query": extended_query, 
        "contexts": contexts, 
        "extracted_issues": extracted_issues if extracted_issues else "없음 (Query_Only 모드)",
        "input_type": input_type,
        "context_count": str(context_count),
        "min_required_docs": str(min_required)})
    
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
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("confirm_answer_prompt")
    structured_llm = llm.with_structured_output(AnswerEnough)
    chain = prompt | structured_llm
    response = chain.invoke({
        "extended_query": extended_query, 
        "contexts": contexts, 
        "extracted_issues": extracted_issues if extracted_issues else "없음 (Query_Only 모드)",
        "answer": answer,
        "context_count": str(context_count),
        "min_required_docs": str(min_required)})
    
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
    
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("retry_answer_prompt")
    structured_llm = llm.with_structured_output(AnswerOutput)
    chain = prompt | structured_llm
    response = chain.invoke({
        "extended_query": extended_query, 
        "contexts": contexts, 
        "extracted_issues": extracted_issues if extracted_issues else "없음 (Query_Only 모드)",
        "previous_answer": previous_answer, 
        "feedback": feedback})
    return response