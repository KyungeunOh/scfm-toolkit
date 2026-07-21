# =========================================================================
# scGPT 어노테이션 툴킷 - 베이스 환경 Dockerfile
#
# 기존 scgpt_env 컨테이너(12일간 수동으로 패키지를 설치해온 상태)를
# 이미지로 "굽어서" 재현 가능하게 만드는 것이 목적.
#
# 베이스: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
#   - 기존 환경과 동일한 베이스. torch/torchvision/torchaudio 이미 포함.
#   - devel 태그를 쓰는 이유: flash-attn 소스 빌드에 nvcc(CUDA 컴파일러)가 필요.
#     runtime 태그에는 nvcc가 없어서 빌드 실패함.
# =========================================================================
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

LABEL maintainer="scgpt-toolkit"
LABEL description="scGPT cell type annotation toolkit - reproducible environment"

# -------------------------------------------------------------------------
# 환경 변수
# -------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WANDB_MODE=offline \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# -------------------------------------------------------------------------
# 시스템 패키지
#   - git: scgpt 등 일부 패키지가 git 의존성을 가질 수 있어 안전하게 포함
#   - build-essential: flash-attn 소스 컴파일에 필요한 C/C++ 툴체인
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        wget \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------
# Python 의존성 설치 (requirements.txt)
#   torch/torchvision/torchaudio는 베이스 이미지가 이미 제공하므로
#   requirements.txt에는 포함하지 않음 (재설치 시 CUDA 빌드가 깨질 위험 방지)
# -------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# torchtext==0.15.2는 PyPI가 아닌 PyTorch 자체 wheel 인덱스에서만 배포됨
# (torch==2.0.1과 짝이 맞는 버전). 일반 PyPI 인덱스에는 존재하지 않으므로
# --index-url로 별도 지정.
#
# 중요: torchtext는 scGPT의 "선택적" 부속 패키지가 아니라 하드 의존성임.
# scgpt/tokenizer/gene_tokenizer.py가 torchtext._torchtext의 Vocab(VocabPybind)을
# 직접 import하므로, 이 단계가 실패하면 import scgpt 자체가 깨짐.
# 따라서 폴백 없이 실패 시 빌드를 중단시킴 (조용히 넘어가면 나중에
# 런타임에서야 깨지는 더 나쁜 상황이 됨).
RUN pip install --no-cache-dir torchtext==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu117 \
    --extra-index-url https://pypi.org/simple

# ninja: flash-attn 소스 컴파일 속도를 크게 높여줌 (없으면 distutils 기본
# 컴파일러로 떨어져서 빌드 시간이 몇 배로 늘어남). requirements.txt가
# 아닌 별도 라인으로 둔 이유: 빌드 전용 도구라 런타임 의존성과 구분.
RUN pip install --no-cache-dir ninja

# -------------------------------------------------------------------------
# flash-attn 설치
#   원본 환경: flash-attn==1.0.4
#   확인 결과: flash-attn 1.x 시절에는 Dao-AILab 공식 GitHub Releases에
#   사전 빌드 wheel이 존재하지 않음 (wheel 배포는 2.x 버전대부터 정착됨).
#   따라서 소스 컴파일이 필수이며, wheel을 찾는 시도는 시간 낭비이므로
#   처음부터 소스 빌드 경로로 진행함.
#
#   주의: 소스 컴파일 시 10~30분 이상 걸릴 수 있음.
#   MAX_JOBS를 제한해 OOM(메모리 부족으로 빌드 프로세스 강제 종료)을 방지.
#   (공식 문서 권장값: MAX_JOBS=4)
# -------------------------------------------------------------------------
ENV MAX_JOBS=4

RUN pip install flash-attn==1.0.4 --no-build-isolation

# -------------------------------------------------------------------------
# 설치 검증 (빌드 단계에서 바로 깨진 환경을 잡아내기 위함)
# -------------------------------------------------------------------------
RUN python -c "import torch; print('torch', torch.__version__, torch.version.cuda)" \
    && python -c "import flash_attn; print('flash_attn ok')" \
    && python -c "import scgpt; print('scgpt ok')" \
    && python -c "from scgpt.tokenizer.gene_tokenizer import GeneVocab; print('scgpt tokenizer (torchtext 의존) ok')" \
    && python -c "import scanpy; print('scanpy', scanpy.__version__)" \
    && python -c "import anndata; print('anndata', anndata.__version__)"

# -------------------------------------------------------------------------
# 애플리케이션 코드 복사 (이후 단계에서 작성할 src/)
# -------------------------------------------------------------------------
COPY src/ /workspace/src/

ENTRYPOINT ["python", "/workspace/src/run.py"]

