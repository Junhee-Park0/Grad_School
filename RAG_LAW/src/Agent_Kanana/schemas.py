from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, ConfigDict

class UserInput(BaseModel):
    """사용자 입력 형식"""
    query : Optional[str] = Field(description = "The user's question")
    document_path : Optional[str] = Field(description = "The path of the document (ex. PDF)")

class InputDocument(BaseModel):
    """입력 문서의 OCR 결과 형식"""
    document : str = Field(..., description = "The parsed document")

class QueryAnswerable(BaseModel):
    """질문이 문서 없이 답변 가능한지 판단하는 형식"""
    answerable : Literal["ANSWERABLE", "NOT_ANSWERABLE"] = Field(description = "Whether the question can be answered without the document")
    reason : str = Field(description = "Reason for the decision")
    
class DocumentIssue(BaseModel):
    """OCR 결과 문서에서 추출한 쟁점 하나의 형식"""
    issue : str = Field(..., description = "The issue extracted from the document")
    position : str = Field(..., description = "The position where the issue is located (e.g. 'Article 10', 'Section 2')")
    reason : str = Field(..., description = "The reason for the issue")
    risk_summary : str = Field(..., description = "The risk summary of the document")

class IssuesList(BaseModel):
    """DocumentIssue의 리스트"""
    issues : List[DocumentIssue] = Field(..., description = "The list of issues extracted from the document")

class CombinedQuery(BaseModel):
    """질문 + 문서 쟁점들 통합한 쿼리 하나의 형식"""
    query : str = Field(..., description = "User's question & document issues")
    type : Literal["Question", "Document"] = Field(..., description = "The type of the query")
    position : Optional[str] = Field(None, description = "The position where the issue is located (e.g. 'Article 10', 'Section 2')")
    reason : Optional[str] = Field(None, description = "The reason for the issue")

    @property
    def to_rag_query(self) -> str:
        """RAG 검색 쿼리로 변환하는 메서드"""
        if self.type == "Question":
            return self.query
        else:
            return f"{self.query} {self.reason}"

class QueryList(BaseModel):
    """CombinedQuery의 리스트"""
    combined_queries : List[CombinedQuery] = Field(..., description = "The combined queries (question + document issues)")

class Metadata(BaseModel):
    """OpenAI와 호환을 위해 정의한 메타데이터 전용 스키마"""
    model_config = ConfigDict(extra="forbid")
    # RAG metadata 필드들
    eff_date: Optional[str] = Field(None, description = "The effective date of the law")
    law_name: Optional[str] = Field(None, description = "The name of the law")
    law_path: Optional[str] = Field(None, description = "The path to the law document")
    # Web search metadata 필드들
    published_date: Optional[str] = Field(None, description = "The published date of the web page")
    query_used: Optional[str] = Field(None, description = "The search query used to find this result")
    domain: Optional[str] = Field(None, description = "The domain of the web page")
    title: Optional[str] = Field(None, description = "The title of the web page")
    editor: Optional[str] = Field(None, description = "The editor of the web page")

class ContextOutput(BaseModel):
    """한 번 더 필터링된 최종 문서 하나의 컨텍스트 형식(RAG + 외부 검색)"""
    rank : Optional[int] = Field(None, description = "Re-calculated rank of the document based on relevance. None before reranking")
    doc_type : Literal["Internal_DB", "External_Web"] 
    text : str = Field(..., description = "The text of the document")
    metadata : Metadata = Field(..., description = "The metadata of the document")
    source : str = Field(..., description = "The source of the document")
    relevance_score : float = Field(..., description = "Re-calculated relevance score of the document")

class ContextList(BaseModel):
    """ContextOutput의 리스트"""
    list_contexts : List[ContextOutput] = Field(..., description = "The list of contexts")

class RAGOutput(BaseModel):
    """RAG 결과 하나의 형식"""
    search_rank : int = Field(..., description = "The rank of the document based on relevance")
    text : str = Field(..., description = "The text of the document")
    source : str = Field(..., description = "The source of the document")
    metadata : Metadata = Field(..., description = "The metadata of the document")
    relevance_score : float = Field(..., description = "The relevance score of the document")

    @property
    def to_context(self) -> ContextOutput:
        return ContextOutput(
            rank = self.search_rank,
            doc_type = "Internal_DB",
            text = self.text,
            metadata = self.metadata,
            source = self.source,
            relevance_score = self.relevance_score
        )

class RAGList(BaseModel):
    """RAGOutput의 리스트"""
    list_rag_results : List[RAGOutput] = Field(..., description = "The list of RAG results")

class EnoughContext(BaseModel):
    """외부 검색 필요성 확인(RAG 결과로 충분한 답변 가능 여부)"""
    enough_context : Literal["ENOUGH", "NOT_ENOUGH"]
    reason : str = Field(..., description = "The reason for the decision (whether or not we need external search)")

class RerankItem(BaseModel):
    """리랭크 결과 하나 — 점수와 순위만 포함 (텍스트는 원본에서 복원)"""
    source : str = Field(..., description = "The source identifier of the document (must match exactly)")
    rank : int = Field(..., description = "Rank after reranking (1 = highest relevance)")
    relevance_score : float = Field(..., description = "Relevance score 0.0~1.0")

class RerankList(BaseModel):
    """RerankItem의 리스트"""
    list_contexts : List[RerankItem] = Field(..., description = "Reranked list with scores and ranks only")

class WebSearchQueries(BaseModel):
    """Web Search를 위한 쿼리 리스트"""
    web_search_queries : List[str] = Field(..., min_items = 1, max_items = 6, description = "The list of web search queries (must be exactly 6 queries)")

class WebSearchOutput(BaseModel):
    """Web Search 결과 하나의 형식"""
    title : str = Field(..., description = "The title of the web page")
    text : str = Field(..., description = "The text of the web page")
    source : str = Field(..., description = "The URL of the web page")
    metadata : Metadata = Field(..., description = "{'title', 'date', 'editor' etc.}")
    relevance_score : float = Field(..., description = "The relevance score of the web page")

    @property
    def to_context(self) -> ContextOutput:
        return ContextOutput(
            rank = None,
            doc_type = "External_Web",
            text = self.text,
            metadata = self.metadata,
            source = self.source,
            relevance_score = self.relevance_score
        )

class WebSearchList(BaseModel):
    """WebSearchOutput의 리스트"""
    list_web_results : List[WebSearchOutput] = Field(..., description = "The list of web search results")

class AnswerOutput(BaseModel):
    """최종 답변 형식"""
    input_type : Literal["Query_Only", "Hybrid", "Error"] 
    answer : str = Field(
        ..., 
        description = "The final answer. MUST end with '## 참고자료' section that lists all cited sources. "
                     "Format: (main content) + '\\n\\n## 참고자료\\n\\n[1] source1\\n[2] source2\\n...'"
    )
    source : List[str] = Field(
        default_factory = list,
        description = "List of source identifiers actually cited in the answer."
    )
    risk_summary : str = Field(default = "", description = "Summary of potential risks, missing info, etc.")
    confidence_score : float = Field(default = 0.0, description = "Internal confidence score for how well the context covers the answer")

class AnswerEnough(BaseModel):
    """최종 답변 적합성 확인"""
    kind : Literal["ENOUGH", "NOT_ENOUGH"] 
    feedback : str = Field(..., description = "The feedback based on the context")
