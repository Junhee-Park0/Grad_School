"""
nodes_kanana_track.py

nodes_kanana.py의 모든 노드를 래핑하여 각 노드의 입·출력을
JSON 파일로 기록하는 추적 레이어.

사용법:
  main_kanana.py 에서 아래처럼 import 경로만 변경하세요.

  # 기존
  from src.Agent_Kanana.nodes_kanana import routing_node, ...

  # 변경 후
  from src.Agent_Kanana.nodes_kanana_track import routing_node, ...

로그 구조:
  logs/
    YYYY-MM-DD_HH-MM-SS/        ← 에이전트 1회 실행 = 폴더 1개
      01_routing_node.json
      02_query_rewriting_node.json
      ...
"""

import os
import json
import functools
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel

# 원본 노드 함수 전부 import
from src.Agent_Kanana.nodes_kanana import (
    routing_node,
    query_rewriting_node,
    document_parsing_node,
    issue_extracting_node,
    rag_searching_node,
    context_evaluating_node,
    web_searching_node,
    context_filtering_node,
    context_reranking_node,
    answer_generating_node,
    answer_evaluating_node,
    answer_regenerating_node,
)

# 로그 기본 경로 (프로젝트 루트 기준)
_BASE_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


# ──────────────────────────────────────────────
# 직렬화 헬퍼
# ──────────────────────────────────────────────

def _serialize(obj: Any) -> Any:
    """Pydantic 모델·dict·list 등을 JSON 직렬화 가능한 형태로 변환."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # 그 외 타입은 문자열로 변환
    return str(obj)


# ──────────────────────────────────────────────
# NodeTracker 클래스
# ──────────────────────────────────────────────

class NodeTracker:
    """
    에이전트 1회 실행마다 타임스탬프 폴더를 생성하고
    각 노드의 입·출력을 순번이 붙은 JSON 파일로 저장한다.
    """

    def __init__(self, base_dir: str = _BASE_LOG_DIR):
        self.base_dir = base_dir
        self.run_dir: Optional[str] = None
        self._counter: int = 0

    def init_run(self) -> str:
        """
        새 실행 시작 — 타임스탬프 이름의 하위 폴더를 생성한다.
        routing_node 진입 시 자동으로 호출된다.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self._counter = 0
        print(f"\n[Tracker] 로그 폴더 생성: {self.run_dir}\n")
        return self.run_dir

    def log(self, node_name: str, input_state: Dict, output_state: Dict) -> None:
        """
        노드 하나의 입·출력을 JSON 파일로 저장한다.

        파일명: {순번:02d}_{node_name}.json
        """
        if self.run_dir is None:
            self.init_run()

        self._counter += 1
        filename = f"{self._counter:02d}_{node_name}.json"
        filepath = os.path.join(self.run_dir, filename)

        log_data = {
            "node":      node_name,
            "order":     self._counter,
            "timestamp": datetime.now().isoformat(),
            "input":     _serialize(input_state),
            "output":    _serialize(output_state),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"[Tracker] {filename} 저장 완료")


# 싱글톤 인스턴스
tracker = NodeTracker()


# ──────────────────────────────────────────────
# 데코레이터
# ──────────────────────────────────────────────

def track_node(fn, *, is_entry: bool = False):
    """
    노드 함수를 감싸는 데코레이터.

    is_entry=True 이면 노드 진입 시 tracker.init_run()을 호출하여
    새 실행 폴더를 생성한다 (routing_node에 사용).
    """
    @functools.wraps(fn)
    def wrapper(state):
        if is_entry:
            tracker.init_run()

        output = fn(state)

        tracker.log(
            node_name=fn.__name__,
            input_state=dict(state),
            output_state=output or {},
        )
        return output

    return wrapper


# ──────────────────────────────────────────────
# 추적이 적용된 노드 함수들
# (main_kanana.py 에서 이 이름 그대로 import 하세요)
# ──────────────────────────────────────────────

routing_node             = track_node(routing_node,             is_entry=True)
query_rewriting_node     = track_node(query_rewriting_node)
document_parsing_node    = track_node(document_parsing_node)
issue_extracting_node    = track_node(issue_extracting_node)
rag_searching_node       = track_node(rag_searching_node)
context_evaluating_node  = track_node(context_evaluating_node)
web_searching_node       = track_node(web_searching_node)
context_filtering_node   = track_node(context_filtering_node)
context_reranking_node   = track_node(context_reranking_node)
answer_generating_node   = track_node(answer_generating_node)
answer_evaluating_node   = track_node(answer_evaluating_node)
answer_regenerating_node = track_node(answer_regenerating_node)
