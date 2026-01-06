from langchain_community.tools import tool
from typing import List, Dict, Optional, Any
from schemas import (UserInput, InputType, InputQuery, InputDocument, 
                     DocumentIssue, IssuesList, RAGOutput, RAGList, 
                     EnoughContext, WebSearchOutput, WebSearchList, 
                     ContextOutput, ContextList, AnswerOutput, AnswerEnough)
from functions import load_prompt, document_ocr as ocr_function, combine_contexts
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

import chromadb
from src.RAG.search_openai_main import NaiveSearchWithAnswer

@tool
def extend_query(original_query : str) -> str:
    """쿼리를 받아서 법률 용어로 확장, 재구성하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("extend_query_prompt")
    chain = prompt | llm
    response = chain.invoke({"original_query": original_query})
    return response.content

@tool
def parse_document_ocr(ocr_result : Any) -> InputDocument:
    """OCR 결과를 파싱하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("parse_document_ocr_prompt")
    structured_llm = llm.with_structured_output(InputDocument)
    chain = prompt | structured_llm
    response = chain.invoke({"document_ocr": ocr_result})
    return response

@tool
def extract_issues(extended_query : str, document : Any, contract_type : str) -> IssuesList:
    """파싱된 문서를 받아와서 쟁점을 추출하는 함수 (질문 들어오면 질문 있는 거랑 관련 있도록 설정)"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("extract_issues_prompt")
    structured_llm = llm.with_structured_output(IssuesList)
    chain = prompt | structured_llm
    response = chain.invoke({
        "query": extended_query, 
        "document": document, 
        "contract_type": contract_type})
    return response

@tool
def search_rag(extended_query : str, issues : Optional[List[DocumentIssue]] = None, rag_method : str = "naive") -> RAGList:
    """RAG 검색 기능을 수행하는 함수 (문서 쟁점이 있으면 함께 검색)"""
    try:
        lawdb_path = "data/Database/LawDB"
        client = chromadb.PersistentClient(path = lawdb_path)
        collection = client.get_or_create_collection("laws")
        
        search_queries = [extended_query]
        if issues:
            for issue in issues:
                issue_query = f"{issue.issue} {issue.reason}"
                search_queries.append(issue_query)
        
        all_results = []
        seen_texts = set() # 이미 가지고 온 문서들
        
        for query in search_queries:
            naive_search_with_answer = NaiveSearchWithAnswer(collection, query)
            formatted_docs = naive_search_with_answer.search()
            
            for doc in formatted_docs:
                text = doc.get("text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(doc)
        
        sorted_docs = sorted(all_results, key = lambda x : x.get("relevance_score", 0.0), reverse = True)
        
        rag_results = []
        for rank, doc in enumerate(sorted_docs):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            score = doc.get("relevance_score", 0.0)
            source = metadata.get("law_path", "")
            output = RAGOutput(
                search_rank = rank,
                text = text,
                source = source,
                metadata = metadata,
                relevance_score = score
            )
            rag_results.append(output)
        
        return RAGList(rag_results = rag_results)
    except Exception as e:
        print(f"Error in search_rag: {e}")
        return RAGList(rag_results = [])

@tool
def enough_context(extended_query : str, rag_results : List[RAGOutput]) -> EnoughContext:
    """RAG 결과로 충분한 답변이 가능한지 여부를 확인하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("enough_context_prompt")
    structured_llm = llm.with_structured_output(EnoughContext)
    chain = prompt | structured_llm
    response = chain.invoke({
        "query": extended_query, 
        "rag_results": rag_results})
    return response

@tool
def search_web(extended_query : str) -> WebSearchList:
    """외부 웹 검색을 수행하는 함수"""
    pass

@tool
def rerank_contexts(extended_query : str, rag_results : RAGList, web_results : WebSearchList) -> ContextList:
    """컨텍스트를 재정렬하는 함수(rank 부분에 값 채워넣기)"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("rerank_contexts_prompt")
    structured_llm = llm.with_structured_output(ContextList)
    chain = prompt | structured_llm
    response = chain.invoke({
        "query": extended_query, 
        "rag_results": rag_results, 
        "web_results": web_results})
    return response

@tool
def generate_answer(extended_query : str, contexts : ContextList, input_type : InputType) -> AnswerOutput:
    """최종 컨텍스트를 받아서 응답을 생성하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("generate_answer_prompt")
    structured_llm = llm.with_structured_output(AnswerOutput)
    chain = prompt | structured_llm
    response = chain.invoke({
        "extended_query": extended_query, 
        "contexts": contexts, 
        "input_type": input_type})
    return response

@tool
def confirm_answer(extended_query : str, contexts : ContextList, answer : AnswerOutput) -> AnswerEnough:
    """최종 답변을 받아서 적합성을 확인하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("confirm_answer_prompt")
    structured_llm = llm.with_structured_output(AnswerEnough)
    chain = prompt | structured_llm
    response = chain.invoke({
        "extended_query": extended_query, 
        "contexts": contexts, 
        "answer": answer})
    return response

@tool
def retry_answer(extended_query : str, contexts : ContextList, previous_answer : AnswerOutput, feedback : AnswerEnough) -> AnswerOutput:
    """최종 답변과 피드백을 받아서 재생성하는 함수"""
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = load_prompt("retry_answer_prompt")
    structured_llm = llm.with_structured_output(AnswerOutput)
    chain = prompt | structured_llm
    response = chain.invoke({
        "extended_query": extended_query, 
        "contexts": contexts, 
        "previous_answer": previous_answer, 
        "feedback": feedback})
    return response