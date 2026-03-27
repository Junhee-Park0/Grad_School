from typing import List, Dict, Optional, Any, Literal
from src.Agent_Kanana.schemas import (UserInput,
                     CombinedQuery, QueryList, RAGOutput, RAGList, 
                     EnoughContext, WebSearchOutput, WebSearchList, 
                     ContextOutput, ContextList, AnswerOutput, AnswerEnough)
import yaml, os
from langchain_core.prompts import ChatPromptTemplate

def load_prompt_kanana(prompt_name: str) -> str:
    """Kanana 프롬프트를 불러오는 함수 - 전체 프롬프트를 문자열로 반환"""
    with open(f"src/Agent_Kanana/prompts_kanana.yaml", "r", encoding = "utf-8") as f:
        prompts = yaml.safe_load(f)
        prompt = prompts.get(prompt_name, {})
    
    # System prompt 구성
    system_prompt = f'{prompt.get("role", "")}\n\n{prompt.get("instructions", "")}'
    
    if prompt.get("constraints"):
        system_prompt += f'\n\n{prompt.get("constraints", "")}'
    
    if prompt.get("criteria"):
        system_prompt += f'\n\n{prompt.get("criteria", "")}'

    # query_categories가 있으면 추가 (generate_search_queries_prompt, revise_search_queries_prompt용)
    if "query_categories" in prompt:
        categories_str = "\n\n## Query Categories:\n"
        for cat in prompt["query_categories"]:
            if isinstance(cat, dict):
                categories_str += f"- {cat.get('category', '')}: {cat.get('description', '')}\n"
                categories_str += f"  예시: {cat.get('example', '')}\n"
            else:
                # yaml에 string 리스트로 정의된 경우
                categories_str += f"- {cat}\n"
        system_prompt += categories_str

    # Human prompt (inputs) 추가
    human_prompt = prompt.get("inputs", "")
    if human_prompt:
        system_prompt += f"\n\n## Inputs:\n{human_prompt}"

    return system_prompt

def determine_input_type(input: UserInput) -> Literal["Query_Only", "Hybrid", "Error"]:
    """UserInput을 받아서 입력 형식을 결정하는 함수"""
    has_document = input.document_path is not None and len(input.document_path) > 0
    has_query = input.query is not None and input.query.strip() != ""
    
    if has_query and not has_document:
        return "Query_Only"
    elif has_query and has_document:
        return "Hybrid"
    else:
        return "Error"

def route_by_input_type(state) -> Literal["Query_Only", "Hybrid", "Error"]:
    """입력(state)을 받아서 입력 형식을 결정하는 함수"""
    return state["input_type"]

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import kss
import re

def document_ocr(document_path: str) -> str:
    """(로컬용) 문서 경로를 받아서 OCR을 수행하는 함수 / 우선은 PDF만 지원"""
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {document_path}")
    
    if not document_path.lower().endswith(".pdf"):
        raise ValueError(f"지원하지 않는 파일 형식입니다: {document_path}")

    refined_full_text = ""
    tesseract_config = "--psm 3 -l kor+eng"
    try:
        with pdfplumber.open(document_path) as f:
            total_pages = len(f.pages)
            # 비어있는 pdf
            if total_pages == 0:
                raise ValueError(f"비어있는 PDF 파일입니다. 다른 파일을 시도해주세요: {document_path}")
            for page_num, page in enumerate(f.pages, 1):
                # 디지털 구분 - chars가 전체 텍스트 길이의 70% 이상이면 디지털로 처리
                extracted_text = page.extract_text()
                is_digital = False
                if extracted_text and extracted_text.strip():
                    if len(page.chars) > len(extracted_text.strip()) * 0.7:
                        is_digital = True
                # 디지털 처리
                if is_digital:
                    current_text = extracted_text
                    source_type = "Digital"
                else:
                    # 디지털 처리 실패 시 OCR 진행
                    print(f"Page {page_num}/{total_pages} performing OCR...")
                    images = convert_from_path(document_path, first_page = page_num, last_page = page_num)
                    if images:
                        current_text = pytesseract.image_to_string(images[0], config = tesseract_config)
                        source_type = "OCR"
                    else:
                        continue
                
                # 문장 구분
                text = re.sub(r'\n{2,}', '[[PARAGRAPH]]', current_text)
                lines = text.split('\n')
                processed_text = ""
                for i in range(len(lines)):
                    line = lines[i].rstrip()
                    if not line:
                        continue
                    if i < len(lines) - 1:
                        next_line = lines[i + 1].strip()
                        is_sentence_end = re.search(r'[.?!함다요임)\]>"\']$', line)
                        is_next_start_marker = re.match(r'^[※*○●□■→\->\=>\d+\.\[]', next_line)
                        if is_sentence_end or is_next_start_marker:
                            processed_text += line + "\n"
                        else:
                            if re.search(r'[가-힣]$', line) and re.match(r'^[가-힣]', next_line):
                                processed_text += line
                            else:
                                processed_text += line + " "
                    else:
                        processed_text += line
                page_processed_text = processed_text.replace("[[PARAGRAPH]]", "\n\n")
                try:
                    page_processed_text = "\n".join(kss.split_sentences(page_processed_text))
                except:
                    pass

                refined_full_text += f"--- [Page {page_num} ({source_type})] ---\n{page_processed_text}\n"
                print(f"Page {page_num}/{total_pages} processed")

        # # OCR 결과 저장 (확인용..)
        # with open(f"data/test/{file_name}.txt", "w", encoding = "utf-8") as f:
        #     f.write(refined_full_text)
        #     print(f"OCR 결과가 data/test/{file_name}.txt에 저장되었습니다.")

    except FileNotFoundError:
        raise  
    except ValueError:
        raise  
    except Exception as e:
        raise RuntimeError(f"PDF 처리 중 예상치 못한 오류 발생: {str(e)}") from e
    
    return refined_full_text

def route_after_document_parsing(state) -> Literal["Hybrid", "Query_Only", "END"]:
    """문서 파싱 후 라우팅 함수"""
    if state.get("input_type") == "Error":
        return "Error"
    elif state.get("parsed_document"):
        return "Hybrid"
    elif state.get("input_type") == "Query_Only" and state.get("doc_parse_failed"):
        return "Query_Only"
    else:
        print(f"⚠️ 예상치 못한 상태: input_type = {state.get('input_type')}, parsed_document = {state.get('parsed_document')}")
        return "Error"

def filter_low_relevance_contexts(contexts: ContextList, min_relevance_score: float = 0.5, max_contexts: int = 12, min_rag_score: float = 0.4) -> ContextList:
    """
    낮은 관련도 컨텍스트를 필터링하는 함수 (토큰 제한 고려)
    Args:
        contexts: 재정렬된 컨텍스트 리스트
        min_relevance_score: 최소 관련도 점수 - 웹용 (기본값: 0.5)
        max_contexts: 유지할 최대 컨텍스트 개수 (기본값: 12, 토큰 제한 고려)
        min_rag_score: 최소 관련도 점수 - RAG용 (기본값: 0.4, 더 관대하게)
    """
    rag_contexts = [context for context in contexts.list_contexts if context.doc_type == "Internal_DB"]
    web_contexts = [context for context in contexts.list_contexts if context.doc_type == "External_Web"]
    # Rerank 전이라면 (아직 relevance_score 스케일이 맞춰지지 않음)
    max_rag_score = max([context.relevance_score for context in rag_contexts], default=0.0) if rag_contexts else 0.0
    
    if max_rag_score < 0.5:
        # Rerank 전: RAG 스케일이 낮으므로 개수 기반으로 보존
        sorted_web = sorted(web_contexts, key = lambda x: (x.relevance_score, -x.rank if x.rank is not None else 0), reverse = True)
        filtered_web = [context for context in sorted_web if context.relevance_score >= min_relevance_score][:10]
        filtered_contexts = rag_contexts[:10] + filtered_web
    else:
        # Rerank 후: 스케일 통일되었으므로 점수 기반 필터링 (RAG는 더 관대하게)
        sorted_rag = sorted(rag_contexts, key=lambda x: (x.relevance_score, -x.rank if x.rank is not None else 0), reverse=True)
        sorted_web = sorted(web_contexts, key=lambda x: (x.relevance_score, -x.rank if x.rank is not None else 0), reverse=True)
        
        # RAG: 0.4 이상, 웹: 0.5 이상
        filtered_rag = [ctx for ctx in sorted_rag if ctx.relevance_score >= min_rag_score]
        filtered_web = [ctx for ctx in sorted_web if ctx.relevance_score >= min_relevance_score]
        
        # 합쳐서 상위 max_contexts개 선택
        combined = filtered_rag + filtered_web
        sorted_combined = sorted(combined, key=lambda x: (x.relevance_score, -x.rank if x.rank is not None else 0), reverse=True)
        filtered_contexts = sorted_combined[:max_contexts]
    
    # 필터링 전후 통계 출력 (RAG vs 웹 검색 결과 분포 확인)
    original_rag_count = sum(1 for ctx in contexts.list_contexts if ctx.doc_type == "Internal_DB")
    original_web_count = sum(1 for ctx in contexts.list_contexts if ctx.doc_type == "External_Web")
    filtered_rag_count = sum(1 for ctx in filtered_contexts if ctx.doc_type == "Internal_DB")
    filtered_web_count = sum(1 for ctx in filtered_contexts if ctx.doc_type == "External_Web")
    
    print(f"필터링 전: RAG {original_rag_count}개, 웹 검색 {original_web_count}개")
    print(f"필터링 후: RAG {filtered_rag_count}개, 웹 검색 {filtered_web_count}개")    
    print(f"📊: {len(contexts.list_contexts)}개 -> {len(filtered_contexts)}개")
    
    return ContextList(list_contexts = filtered_contexts)

def truncate_context_texts(contexts: ContextList, max_text_length: int = 2000) -> ContextList:
    """컨텍스트 텍스트 길이를 제한하는 유틸리티 함수
    
    Args:
        contexts: 제한할 컨텍스트 리스트
        max_text_length: 최대 텍스트 길이 (기본값: 2000자)
    
    Returns:
        텍스트가 제한된 컨텍스트 리스트
    """
    truncated_contexts = []
    for context in contexts.list_contexts:
        if len(context.text) > max_text_length:
            truncated_context = ContextOutput(
                rank = context.rank,
                doc_type = context.doc_type,
                text = context.text[:max_text_length] + "... (텍스트가 길어 일부만 표시)",
                metadata = context.metadata,
                source = context.source,
                relevance_score = context.relevance_score
            )
            truncated_contexts.append(truncated_context)
        else:
            truncated_contexts.append(context)
    return ContextList(list_contexts=truncated_contexts)

def route_by_enough_context(state) -> Literal["ENOUGH", "NOT_ENOUGH"]:
    """외부 검색 필요성 확인 결과를 받아서 노드 이름을 반환하는 함수"""
    enough_context = state.get("enough_context")
    if enough_context and enough_context.enough_context == "ENOUGH":
        return "ENOUGH"
    else:
        return "NOT_ENOUGH"

def route_by_enough_answer(state) -> Literal["ENOUGH", "NOT_ENOUGH"]:
    """최종 답변 적합성 확인 결과를 받아서 노드 이름을 반환하는 함수"""
    answer_enough = state.get("answer_enough")
    if answer_enough and answer_enough.kind == "ENOUGH":
        return "ENOUGH"
    else:
        return "NOT_ENOUGH"

def should_regenerate(state) -> Literal["YES", "NO"]:
    """
    답변 재생성 필요성 확인 함수
    """
    current_retry = state.get("answer_retry_count", 0)
    
    # 최대 3번까지 재생성 허용 (0, 1, 2번째 재생성)
    if current_retry < 3:
        print(f"🔄 재생성 허용 (현재 시도: {current_retry}/3)")
        return "YES"
    else:
        print(f"⛔ 재생성 횟수 초과 (현재 시도: {current_retry}/3)")
        return "NO"


