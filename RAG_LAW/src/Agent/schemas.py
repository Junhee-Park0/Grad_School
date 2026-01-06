from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    """사용자 입력 형식"""
    query : Optional[str] = Field(description = "The user's question")
    document_path : Optional[List[str]] = Field(description = "The path of the documents (ex. PDF)")

class InputType(BaseModel):
    """입력 형식"""
    input_type : Literal["Query_Only", "Document_Only", "Hybrid", "Error"]

class InputQuery(BaseModel):
    """입력 쿼리/문서의 형식"""
    query : str = Field(..., description = "The user's question")
    enhanced_query : str = Field(..., description = "The enhanced, expanded question")
    document_ocr : Optional[dict] = Field(description = "The OCR result of the document")
    input_type : InputType

class InputDocument(BaseModel):
    """입력 문서의 OCR 결과 형식"""
    document : str = Field(..., description = "The parsed document")
    contract_type : str = Field(..., description = "The type of the contract")

class DocumentIssue(BaseModel):
    """OCR 결과 문서에서 추출한 쟁점 하나의 형식"""
    issue : str = Field(..., description = "The issue extracted from the document")
    position : str = Field(..., description = "The position where the issue is located (e.g. 'Article 10', 'Section 2')")
    reason : str = Field(..., description = "The reason for the issue")
    risk_summary : str = Field(..., description = "The risk summary of the document")

class IssuesList(BaseModel):
    """DocumentIssue의 리스트"""
    issues : List[DocumentIssue] = Field(..., description = "The list of issues extracted from the document")

class RAGOutput(BaseModel):
    """RAG 결과 하나의 형식"""
    search_rank : int = Field(..., description = "The rank of the document based on relevance")
    text : str = Field(..., description = "The text of the document")
    source : str = Field(..., description = "The source of the document")
    metadata : dict = Field(..., description = "The metadata of the document")
    relevance_score : float = Field(..., description = "The relevance score of the document")

class RAGList(BaseModel):
    """RAGOutput의 리스트"""
    list_rag_results : List[RAGOutput] = Field(..., description = "The list of RAG results")

class EnoughContext(BaseModel):
    """외부 검색 필요성 확인(RAG 결과로 충분한 답변 가능 여부)"""
    enough_context : Literal["ENOUGH", "NOT_ENOUGH"]
    reason : str = Field(..., description = "The reason for the decision (whether or not we need external search)")

class WebSearchOutput(BaseModel):
    """Web Search 결과 하나의 형식"""
    title : str = Field(..., description = "The title of the web page")
    text : str = Field(..., description = "The text of the web page")
    source : str = Field(..., description = "The URL of the web page")
    metadata : dict = Field(..., description = "{'title', 'date', 'editor' etc.}")

class WebSearchList(BaseModel):
    """WebSearchOutput의 리스트"""
    list_web_results : List[WebSearchOutput] = Field(..., description = "The list of web search results")

class ContextOutput(BaseModel):
    """한 번 더 필터링된 최종 문서 하나의 컨텍스트 형식(RAG + 외부 검색)"""
    rank : Optional[int] = Field(None, description = "Re-calculated rank of the document based on relevance. None before reranking")
    doc_type : Literal["Internal_DB", "External_Web"] 
    text : str = Field(..., description = "The text of the document")
    metadata : dict = Field(..., description = "The metadata of the document")
    source : str = Field(..., description = "The source of the document")
    relevance_score : float = Field(..., description = "Re-calculated relevance score of the document")

class ContextList(BaseModel):
    """ContextOutput의 리스트"""
    list_contexts : List[ContextOutput] = Field(..., description = "The list of contexts")

class AnswerOutput(BaseModel):
    """최종 답변 형식"""
    input_type : InputType 
    answer : str = Field(..., description = "The final answer")
    source : List[str] = Field(..., description = "List of sources that were actually used when generating the answer.")
    context : List[ContextOutput] = Field(..., description = "The total contexts that were actually used when generating the answer")
    risk_summary : str = Field(..., description = "Summary of potential risks, missing info, etc.")
    retry_count : int = Field(0, description = "Number of retries for a satisfactory answer")
    confidence_score : float = Field(default = 0.0, description = "Internal confidence score for how well the context covers the answer")

class AnswerEnough(BaseModel):
    """최종 답변 적합성 확인"""
    kind : Literal["ENOUGH", "NOT_ENOUGH"] 
    feedback : str = Field(..., description = "The feedback based on the context")
