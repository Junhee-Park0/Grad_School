from typing import List, Dict, Optional, Any
from schemas import (UserInput, InputType, InputQuery, InputDocument, 
                     DocumentIssue, IssuesList, RAGOutput, RAGList, 
                     EnoughContext, WebSearchOutput, WebSearchList, 
                     ContextOutput, ContextList, AnswerOutput, AnswerEnough)
import yaml
from langchain_core.prompts import ChatPromptTemplate

def load_prompt(prompt_name : str) -> str:
    """프롬프트를 불러오는 함수"""
    with open(f"src/Agent/prompts.yaml", "r", encoding = "utf-8") as f:
        prompts = yaml.safe_load(f)
        prompt = prompts.get(prompt_name, {})

    system_prompt = f"{prompt.get("role", "")}\n\n{prompt.get("instructions", "")}\n\n{prompt.get("constraints", "")}\n\n{prompt.get("criteria", "")}"
    human_prompt = prompt.get("inputs", "")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    return prompt_template

def route_input(input : UserInput) -> InputType:
    """입력을 받아서 입력 형식을 결정하는 함수"""
    has_document = input.document_path is not None and len(input.document_path) > 0
    has_query = input.query is not None and input.query.strip() != ""
    
    if has_query and not has_document:
        return InputType(input_type = "Query_Only")
    elif has_document and not has_query:
        return InputType(input_type = "Document_Only")
    elif has_query and has_document:
        return InputType(input_type = "Hybrid")
    else:
        return InputType(input_type = "Error")

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import kss
import re

def document_ocr(document_path : str, file_name : str) -> str:
    """(로컬용) 문서 경로를 받아서 OCR을 수행하는 함수 / 우선은 PDF만 지원"""
    if not document_path.endswith(".pdf"):
        return "Error: Unsupported file type"

    refined_full_text = ""
    tesseract_config = "--psm 3 -l kor+eng"
    try:
        with pdfplumber.open(document_path) as f:
            total_pages = len(f.pages)
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

        # OCR 결과 저장 (확인용..)
        with open(f"data/test/{file_name}.txt", "w", encoding = "utf-8") as f:
            f.write(refined_full_text)
            print(f"OCR 결과가 data/test/{file_name}.txt에 저장되었습니다.")

    except Exception as e:
        return f"Error: {e}"
    
    return refined_full_text


def combine_contexts(rag_results : List[RAGOutput], web_search_results : List[WebSearchOutput]) -> List[ContextOutput]:
    """RAG 결과와 웹 검색 결과를 통합하는 함수"""
    combined_contexts = []
    
    for rag_result in rag_results:
        context = ContextOutput(
            rank = None,  
            doc_type = "Internal_DB",
            text = rag_result.text,
            metadata = rag_result.metadata,
            source = rag_result.source,
            relevance_score = rag_result.relevance_score
        )
        combined_contexts.append(context)
    
    for web_result in web_search_results:
        context = ContextOutput(
            rank = None,  
            doc_type = "External_Web",
            text = web_result.text,
            metadata = web_result.metadata,
            source = web_result.source,
            relevance_score = 0.0  
        )
        combined_contexts.append(context)
    
    return combined_contexts
