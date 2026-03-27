import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 프로젝트 설정에서 KANANA 경로를 가져옵니다.
# (search_kanana_main.py에서와 동일하게 사용)
from utils.config import Config


def print_gpu_info():
    """GPU 유무 및 PyTorch에서 인식 여부 출력"""
    print("=== GPU / CUDA 정보 체크 ===")
    print(f"torch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능 여부 (torch.cuda.is_available()): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA 디바이스 개수: {num_devices}")
        for i in range(num_devices):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f" - GPU {i}: {name}, compute capability: {cap}")
    else:
        print("PyTorch에서 CUDA를 인식하지 못했습니다. (GPU가 없거나, 드라이버/설치 문제일 수 있습니다.)")

    print("=" * 40, "\n")


def load_model_cpu_only(model_path: str):
    """CPU에서만 모델 로딩 & 간단 생성 테스트"""
    print("=== 1단계: CPU 모델 로딩 테스트 ===")
    print(f"사용할 model_path: {model_path}")

    start = time.time()
    print("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    print(f"토크나이저 로딩 완료 (경과 시간: {time.time() - start:.2f}초)")

    print("모델 로딩(quantization 없음, CPU) 중...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,  # 가장 안전한 설정
    )
    print(f"모델 로딩 완료 (경과 시간: {time.time() - start:.2f}초)")

    # 간단 생성 테스트
    prompt = "안녕하세요. 간단한 자기소개를 해 주세요."
    inputs = tokenizer(prompt, return_tensors="pt")

    print("CPU에서 간단 생성 테스트 중...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    print(f"생성 완료 (경과 시간: {time.time() - start:.2f}초)")

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== 생성 결과 ===")
    print(text)
    print("=" * 40, "\n")


def load_model_gpu_if_available(model_path: str):
    """CUDA가 가능하면 GPU로 모델 로딩 & 간단 생성 테스트 (4bit 아님)"""
    print("=== 2단계: GPU 모델 로딩 테스트 (가능한 경우만) ===")

    if not torch.cuda.is_available():
        print("CUDA 사용 불가 → GPU 로딩 테스트는 건너뜁니다.")
        print("=" * 40, "\n")
        return

    device = "cuda"
    print(f"디바이스: {device}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

    print("토크나이저 로딩 중...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    print(f"토크나이저 로딩 완료 (경과 시간: {time.time() - start:.2f}초)")

    print("모델 로딩(4bit 아님, GPU) 중...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,               # GPU에 올리기
        torch_dtype=torch.float16,       # GPU에서 보통 사용하는 dtype
    )
    print(f"모델 로딩 완료 (경과 시간: {time.time() - start:.2f}초)")

    prompt = "안녕하세요. GPU에서 잘 동작하는지 테스트 중입니다."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("GPU에서 간단 생성 테스트 중...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    print(f"생성 완료 (경과 시간: {time.time() - start:.2f}초)")

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== 생성 결과 ===")
    print(text)
    print("=" * 40, "\n")


def main():
    model_path = Config.KANANA_MODEL_PATH
    print(f"Config.KANANA_MODEL_PATH = {model_path}")
    if not os.path.exists(model_path):
        print("⚠️  모델 경로가 존재하지 않습니다. 경로 설정을 먼저 확인하세요.")
        return

    print_gpu_info()

    # 1단계: CPU에서 먼저 모델이 정상적으로 뜨는지 확인
    try:
        load_model_cpu_only(model_path)
    except Exception as e:
        print("❌ CPU 모델 로딩 중 에러 발생:")
        print(e)
        return

    # 2단계: CUDA가 가능하다면 GPU 로딩도 시도
    try:
        load_model_gpu_if_available(model_path)
    except Exception as e:
        print("❌ GPU 모델 로딩/생성 중 에러 발생:")
        print(e)


if __name__ == "__main__":
    main()