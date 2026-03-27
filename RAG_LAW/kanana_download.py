def download_kanana():
    """
    Hugging Face에서 Kanana 모델을 다운로드하는 함수
    (최초 1회만 실행)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    model_name = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
    save_dir = "E:\Kanana_Model"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 온라인에서 모델 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 모델 저장 (지정한 경로에 저장)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print(f"✅모델과 토크나이저가 {save_dir}에 저장되었습니다.")

if __name__ == "__main__":
    download_kanana()

