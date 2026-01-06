import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

class ParsingAndChunking():
    def __init__(self):
        pass
    
    def read_json_file(self, law_json_path):
        """파일 경로 받아서 파일 열기"""
        with open(law_json_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
        return data
    
    def norm_date(self, date):
        """2025006 -> 2025-09-06"""
        if not date:
            return None
        return f"{date[:4]}-{date[4:6]}-{date[6:]}"

    def norm_hang_no(self, hang_num):
        """'①' -> '1', '②' -> '2', 그대로/None 허용"""
        # 동그리 숫자들 정리하기
        CIRCLED = dict(zip("①②③④⑤⑥⑦⑧⑨⑩", map(str, range(1,11))))
        if not hang_num: 
            return None
        return CIRCLED.get(hang_num, hang_num)

    def norm_ho_no(self, ho_num):
        """'5.' -> '5'"""
        if not ho_num:
            return None
        return ho_num.rstrip(".").strip()

    def norm_mock_no(self, mock_num):
        """'가.' -> '1'"""
        GANADA = {f"{num}." : str(i) for i, num in enumerate("가나다라마바사아자차카타파하", start = 1)}
        if not mock_num:
            return None
        return GANADA.get(mock_num, mock_num)

    def build_chunk_id(self, law_id, junmun_key, text_type, branch_no = None, hang_no = None, ho_no = None, mock_no = None):
        """
        이번 chunk의 전체 제목 정하기
        text_type : 전문/~조
        최종 예시) LAW-011357-제 1장-3조-2항-5호...
        """
        base = f"LAW-{law_id}-{junmun_key}-{text_type}"
        if branch_no is not None:
            base += f"({branch_no})"
        if hang_no is not None:
            base += f"-{hang_no}항"
        if ho_no is not None:
            base += f"-{ho_no}호"
        if mock_no is not None:
            base += f"-{mock_no}목"    
        return base
    
    def as_list(self, x):
        if not x:
            return []
        return x if isinstance(x, list) else [x]

    def clean_text(self, text):
        if not text:
            return ""
        if isinstance(text, list):
            text = " ".join(map(str, text))
        # <개정 ~> 부분 없애기
        text = re.sub(r'<[^>]+>', "", text) 
        # [ ] -> ( )
        text = text.replace("[", "("). replace("]", ")")
        # \ 없애기
        text = text.replace("\\", "")
        #「 」-> " "
        text = text.replace("「", '"').replace("」", '"')
        # 요상한 공백들 없애기
        text = text.replace("\u3000", " ").replace("\xa0", " ")
        # 공백 여러 개, 탭 -> 공백 하나로
        text = re.sub(r"[ \t]+", " ", text)
        # 빈 줄 여러 개 -> 엔터 한 번으로
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def get_basic_information(self, data):
        ### 기본 정보 - law_information
        law_information = {
        # 법령키
        'law_key' : data['법령']['법령키'],
        # 기본정보
        'law_name' : data['법령']['기본정보']['법령명_한글'],
        'law_id' : data['법령']['기본정보']['법령ID'],
        'ministry' : data['법령']['기본정보']['소관부처']['content'],
        'eff_date' : data['법령']['기본정보']['시행일자'],
        'prom_date' : data['법령']['기본정보']['공포일자']}

        return law_information
    
    def get_jomun_information(self, data, law_id, law_information):
        chunks = []

        ### 추가 : 법률 이름 저장해두기
        embed_law_name = law_information.get("law_name", "")   
        ### 추가 : 전문 정보 저장
        current_pyeon = ""
        current_jang = "" # 현재의 전문 (~절일 수도)
        junmun_content = "" # 전체 완성된 전문
        # 전문 없는 경우 
        junmun_num = ""
        junmun_key = ""

        # 조문 
        for jomun in data['법령']['조문']['조문단위']:
            jomun_num = jomun['조문번호']
            jomun_key = jomun['조문키']
            jomun_date = self.norm_date(jomun['조문시행일자'])
            text_type = jomun['조문여부'] # 조문/전문
            
            jomun_content = self.clean_text(jomun['조문내용']) # 전문이든 조문이든 (전문인 경우 continue)

            if text_type == "전문":
                if embed_law_name == "자본시장과 금융투자업에 관한 법률":
                    match_pyeon = re.match(r"^(제.+편)", jomun_content.strip())
                    match_jang = re.match(r"^(제.+장)", jomun_content.strip())
                    is_pyeon = bool(match_pyeon)
                    is_jang = bool(match_jang)
                    # ~편인지 확인
                    if is_pyeon:
                        current_pyeon = jomun_content
                        current_jang = "" # 장 리셋
                        junmun_content = current_pyeon
                        junmun_num = match_pyeon.group(1)
                        junmun_key = jomun_content[:3]
                    # ~장인지 확인
                    elif is_jang:
                        current_jang = jomun_content
                        junmun_content = f"{current_pyeon} {current_jang}"
                    else:
                        junmun_content = f"{current_pyeon} {current_jang} {jomun_content}"
                        
                else:
                    match = re.match(r"^(제.+장)", jomun_content.strip())
                    is_jang = bool(match)
                     # ~장인지 확인
                    if is_jang:
                        junmun_content = jomun_content
                        current_jang = jomun_content
                        junmun_num = match.group(1)
                        junmun_key = jomun_content[:3]
                    else:
                        # ~장 + ~절 붙여서
                        junmun_content = f"{current_jang} {jomun_content}"
                continue

            jomun_branch = jomun.get("조문가지번호", 0)
            is_branch = bool(jomun_branch)
            
            # 조문 자체의 정보 저장
            base_meta = {
                "law_meta" : law_information,
                "junmun_num" : junmun_num,
                "jomun_key" : jomun_key,
                "jomun_num" : jomun_num,
                "jomun_date" : jomun_date,
                "text_type" : text_type,
                "is_branch" : is_branch,
                "jomun_branch" : jomun_branch
            }

            jomun_with_jo = f"{jomun_num}조" if jomun_num else "조문"
            branch_with_gaji = f"(가지{jomun_branch})" if is_branch else ""
            jomun_with_branch = f"{jomun_with_jo}{branch_with_gaji}"

            ### 추가 : 법률 - 전문
            emb_junmun_context = f"{embed_law_name} {junmun_content}"
            
            # 조문 - 항 있음
            hangs = self.as_list(jomun.get("항"))
            hangs_cnt = len(hangs) # 이번 조문에 항이 몇 개인지

            ### 추가 : 법률 - 전문 - 조문
            emb_jomun_context = f"{emb_junmun_context} {jomun_content}"

            # 항 없이 조문만 있는 경우
            if not hangs:
                chunk = {
                    **base_meta,
                    "chunk_id" : self.build_chunk_id(law_id, junmun_key, jomun_with_jo, jomun_branch),
                    "section_type" : "조문",
                    "path" : jomun_with_branch,
                    "hangs_cnt" : 0,
                    "original_text" : jomun_content,
                    "embedding_text" : emb_jomun_context
                }
                chunks.append(chunk)
                continue

            for hang in hangs:
                # 항 정보부터 가져오기
                hang_no_raw = hang.get("항번호")
                hang_no = self.norm_hang_no(hang_no_raw)
                hang_text = self.clean_text(hang.get("항내용"))

                ### 추가 : 법률 - 전문 - 조문 - (항)
                emb_hang_context = f"{emb_jomun_context} {hang_text}" if hang_text else emb_jomun_context

                # 호 정보 가져오기
                hos = self.as_list(hang.get("호"))
                hos_cnt = len(hos)
                is_no_hang = bool(hos) and not (hang.get("항번호") or hang.get("항내용")) # 항 x 

                # 1. [항 | ... ]의 구조 (항의 내용이 없는 경우)
                if is_no_hang:
                    for ho in hos:
                        ho_no_raw = ho.get("호번호")
                        ho_no = self.norm_ho_no(ho_no_raw)
                        ho_text = self.clean_text(ho.get("호내용"))

                        ### 추가 : 법률 - 전문 - 조문 - (항) - (호) -> 항은 없음
                        emb_ho_context = f"{emb_hang_context} {ho_text}" if ho_text else emb_hang_context

                        # 호와 목도 마찬가지.. 
                        mocks = self.as_list(ho.get("목"))
                        mocks_cnt = len(mocks)

                        # [항 | 호]의 구조
                        if ho_text:
                            chunk = {
                                **base_meta,
                                "chunk_id" : self.build_chunk_id(law_id, junmun_key, jomun_with_jo, jomun_branch, hang_no = None, ho_no = ho_no),
                                "section_type" : "호",
                                "path" : f"{junmun_content} - {jomun_with_branch} - {ho_no}호",
                                "ho_no" : ho_no,
                                "hangs_cnt" : hangs_cnt,
                                "hos_cnt" : hos_cnt,
                                "original_text" : ho_text,
                                "embedding_text" : emb_ho_context,
                                "parent": {                            
                                "jomun_label": jomun_with_jo,
                                "junmun_text" : junmun_content,
                                "jomun_text": jomun_content,
                                "branch_no": jomun_branch
                                }
                            }
                            chunks.append(chunk)

                        # [항 | 호 - 목]의 구조
                        for mock in mocks:
                            mock_no_raw = mock.get("목번호")
                            mock_no = self.norm_mock_no(mock_no_raw)
                            mock_text = self.clean_text(mock.get("목내용"))

                            ### 추가 : 법률 - 전문 - 조문 - (항) - (호) - (목) -> 항은 없음
                            emb_mock_context = f"{emb_ho_context} {mock_text}" if mock_text else emb_ho_context

                            if mock_text:
                                chunk = {
                                    **base_meta,
                                    "chunk_id" : self.build_chunk_id(law_id, junmun_key, jomun_with_jo, jomun_branch, hang_no = None, ho_no = ho_no, mock_no = mock_no),
                                    "section_type" : "목",
                                    "path" : f"{junmun_content} - {jomun_with_branch} - {ho_no}호 - {mock_no}목",
                                    "ho_no" : ho_no,
                                    "mock_no" : mock_no,
                                    "hangs_cnt" : hangs_cnt,
                                    "hos_cnt" : hos_cnt,
                                    "mocks_cnt" : mocks_cnt,
                                    "original_text" : mock_text,
                                    "embedding_text" : emb_mock_context,
                                    "parent": {                            
                                    "jomun_label": jomun_with_jo,
                                    "junmun_text" : junmun_content,
                                    "jomun_text": jomun_content,
                                    "ho_text" : ho_text,
                                    "branch_no": jomun_branch
                                    }
                                }
                                chunks.append(chunk)
                    continue
                
                # [항]에서 끝난다면
                if hang_text:
                    chunk = {
                        **base_meta,
                        "chunk_id" : self.build_chunk_id(law_id, junmun_key, jomun_with_jo, jomun_branch, hang_no),
                        "section_type" : "항",
                        "path" : f"{junmun_content} - {jomun_with_branch} - {hang_no}항",
                        "hang_no" : hang_no,
                        "hangs_cnt" : hangs_cnt,
                        "original_text" : hang_text,
                        "embedding_text" : emb_hang_context,
                        "parent": {                            
                        "jomun_label": jomun_with_jo,
                        "junmun_text" : junmun_content,
                        "jomun_text": jomun_content,
                        "branch_no": jomun_branch
                        }
                    }
                    chunks.append(chunk)

                if not hos:
                    continue

                for ho in hos:
                    ho_no_raw = ho.get("호번호")
                    ho_no = self.norm_ho_no(ho_no_raw)
                    ho_text = self.clean_text(ho.get("호내용"))

                    ### 추가 : 법률 - 전문 - 조문 - (항) - (호) 
                    emb_ho_context = f"{emb_hang_context} {ho_text}" if ho_text else emb_hang_context
                    
                    mocks = self.as_list(ho.get("목"))
                    mocks_cnt = len(mocks)

                    # [항 - 호]의 구조
                    if ho_text:
                        chunk = {
                            **base_meta,
                            "chunk_id" : self.build_chunk_id(law_id, junmun_key, jomun_with_jo, jomun_branch, hang_no, ho_no),
                            "section_type" : "호",
                            "path" : f"{junmun_content} - {jomun_with_branch} - {hang_no}항 - {ho_no}호",
                            "hang_no": hang_no,
                            "ho_no" : ho_no,
                            "hangs_cnt" : hangs_cnt,
                            "hos_cnt" : hos_cnt,
                            "original_text" : ho_text,
                            "embedding_text" : emb_ho_context,
                            "parent": {                            
                            "jomun_label": jomun_with_jo,
                            "junmun_text" : junmun_content,
                            "jomun_text": jomun_content,
                            "hang_text" : hang_text,
                            "branch_no": jomun_branch
                            }
                        }
                        chunks.append(chunk)

                    if not mocks:
                        continue

                    # [항 - 호 - 목]의 구조
                    for mock in mocks:
                        mock_no_raw = mock.get("목번호")
                        mock_no = self.norm_mock_no(mock_no_raw)
                        mock_text = self.clean_text(mock.get("목내용"))

                        ### 추가 : 법률 - 전문 - 조문 - (항) - (호) - (목)
                        emb_mock_context = f"{emb_ho_context} {mock_text}" if mock_text else emb_ho_context

                        if mock_text:
                            chunk = {
                                **base_meta,
                                "chunk_id" : self.build_chunk_id(law_id, junmun_key, jomun_with_jo, jomun_branch, hang_no, ho_no, mock_no),
                                "section_type" : "목",
                                "path" : f"{junmun_content} - {jomun_with_branch} - {hang_no}항 - {ho_no}호 - {mock_no}목",
                                "hang_no": hang_no,
                                "ho_no" : ho_no,
                                "mock_no" : mock_no,
                                "hangs_cnt" : hangs_cnt,
                                "hos_cnt" : hos_cnt,
                                "mocks_cnt" : mocks_cnt,
                                "original_text" : mock_text,
                                "embedding_text" : emb_mock_context,
                                "parent": {                            
                                "jomun_label": jomun_with_jo,
                                "junmun_text" : junmun_content,
                                "jomun_text": jomun_content,
                                "hang_text" : hang_text,
                                "ho_text" : ho_text,
                                "branch_no": jomun_branch
                                }
                            }
                            chunks.append(chunk)

        return chunks
    
    def chunking(self, chunks, max_len, child_max_len):
        """
        Input : 파싱 완료된 chunks
        - 각 chunk의 길이 측정 
        - 'embedding_text'가 max_len를 넘는 경우에는...
            - parent, child 분할
            - parent, child 문장 단위로 분할하기
            - child가 child_max_len자 넘지 않도록 청킹하기
            - parent 부분으로 나머지 채우기 (max_len를 넘지 않게, 대신 뒷 맥락부터)
            - 나머지 부분 똑같이 가져오고, 'embedding_text'만 바꿔서 새롭게 저장
            - 'chunk_parts' : 몇 개로 나누었는지
            - 'chunk_no' : 몇 번째 청크인지 
        Output : 청킹까지 완료된 chunks
        """
        output = []
        for chunk in chunks:
            chunk_len = len(chunk.get("embedding_text"))

            # max_len 안 넘으면
            if chunk_len <= max_len:
                new_chunk = {**chunk}
                new_chunk["text_len"] = chunk_len
                new_chunk["chunk_parts"] = 1
                new_chunk["chunk_no"] = 1
                output.append(new_chunk)

            else:
                original_text = chunk.get("original_text", "")
                embedding_text = chunk.get("embedding_text", "")

                if not embedding_text.endswith(original_text):
                    print(f"오류 : {chunk.get('chunk_id')} : embedding_text가 original_text로 끝나지 않습니다.")
                    output.append(chunk)
                    continue

                parent = embedding_text[:-len(original_text)].rstrip()
                child = original_text

                # child가 길면
                if len(child) > child_max_len:
                    child_merged = self.split_sentences(child)
                    child_packs = self.pack_sentences(child_merged, child_max_len)

                # child 안 길면 (parent가 긺)
                else:   
                    child_packs = [child]

                packs_num = len(child_packs)

                # parent 쪼개기
                parents_merged = self.split_sentences(parent)

                for i, child_pack in enumerate(child_packs, 1):
                    parent_budget = max_len - len(child_pack) - 1

                    if parent_budget < 0:
                        parent_budget = 0
                    
                    parent_pack = []
                    parent_pack_len = 0

                    for parent_sent in reversed(parents_merged):
                        parent_sent = parent_sent.strip()
                        if not parent_sent:
                            continue

                        parent_sent_len = len(parent_sent)
                        expected_len = parent_pack_len + parent_sent_len + (1 if parent_pack else 0) # 처음은 0, 그 다음부터는 띄어쓰기

                        if expected_len > parent_budget:
                            break 

                        parent_pack.insert(0, parent_sent)
                        parent_pack_len = expected_len
                    
                    final_parent_pack = " ".join(parent_pack)
                    final_pack = f"{final_parent_pack} {child_pack}"

                    new_chunk = {**chunk}
                    new_chunk["original_text"] = child_pack
                    new_chunk["embedding_text"] = final_pack
                    new_chunk["text_len"] = len(final_pack)
                    new_chunk["chunk_parts"] = packs_num
                    new_chunk["chunk_no"] = i

                    original_chunk_id = new_chunk.get("chunk_id", "chunk")
                    new_chunk["chunk_id"] = f"{original_chunk_id}_chunk{i}"

                    output.append(new_chunk)
                

        return output

    def split_sentences(self, text) -> List[str]:
        """텍스트 문장 단위로 나누기 
        -> List["안녕하세요", "만나서 반가워요"]
        - 기준 : 마침표, 물음표, 느낌표 등 + 숫자 리스트 (1. ), 한글 목차 (가. ), 원문자(①)
        """
        if not text:
            return []
        if isinstance(text, list):
            s = " ".join(map(str, text))
        else:
            s = str(text)
        # 공백 제거
        s = " ".join(s.split())

        pattern = r'(?:(?<=[.!?])(?<!\d\.)\s+)|(?:\s+(?=(?:\d+\.|[가-하]\.|[①-⑮])\s))'
        SENT_SEP = re.compile(pattern)
        parts = SENT_SEP.split(s)
        parts = [p.strip() for p in parts if p and p.strip()]

        # '다만'으로 시작하는 문장의 경우, 앞 문장까지 끌어오기
        merged = []
        for sent in parts:
            if sent.startswith("다만") and merged:
                merged[-1] = merged[-1] + " " + sent
            else:
                merged.append(sent)
        return merged

    def pack_sentences(self, merged, max_len):
        """문장 단위로 분할된 문장 리스트를 길이에 맞게 패킹"""
        packs = []
        pack = []
        pack_len = 0

        for sent in merged:
            sent = sent.strip()
            sent_len = len(sent) + (1 if pack else 0)

            if pack_len + sent_len <= max_len:
                pack.append(sent)
                pack_len += sent_len
            else:
                if pack:
                    packs.append(" ".join(pack))
                if len(sent) <= max_len:
                    pack = [sent]
                    pack_len = len(sent)
                else:
                    while len(sent) > max_len:
                        packs.append(sent[:max_len])
                        sent = sent[max_len:]
                    if sent:
                        pack = [sent]
                        pack_len = len(sent)
                    else:
                        pack = []
                        pack_len = 0
        if pack:
            packs.append(" ".join(pack))

        return packs

    def save_json_file(self, output, file_name):
        file_path = f'data/Laws/Parsed/{file_name}_parsed.json'
        with open(file_path, "w", encoding = 'utf-8') as f:
            json.dump(output, f, ensure_ascii = False, indent = 4)


    def parse_and_chunk(self, folder_name, max_len, child_max_len):
        law_jsons_folder = Path(f"data/Laws/Raw/{folder_name}")
        law_json_paths = list[Path](law_jsons_folder.glob("*.json"))
        laws_parsed_chunked = []

        for law_json_path in law_json_paths:
            data = self.read_json_file(law_json_path)
            law_information = self.get_basic_information(data)
            law_id = law_information['law_key']
            law_name = law_information['law_name']
            chunks = self.get_jomun_information(data, law_id, law_information)
            output = self.chunking(chunks, max_len, child_max_len)
            self.save_json_file(output, law_name)
            print(f"{law_name} 파싱 완료 !")
            laws_parsed_chunked.extend(output)

        return laws_parsed_chunked

if __name__ == "__main__":
    p_c = ParsingAndChunking()
    laws_outputs = p_c.parse_and_chunk("data/Laws/Raw/Laws", max_len = 250, child_max_len = 200)

    with open("data/Laws/Processed/laws_parsed.json", "w", encoding = 'utf-8') as f:
        json.dump(laws_outputs, f, ensure_ascii = False, indent = 4)