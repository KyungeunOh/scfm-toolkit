# =========================================================================
# scGPT 어노테이션 툴킷 - 베이스 환경 Dockerfile
#
# 기존 scgpt_env 컨테이너(12일간 수동으로 패키지를 설치해온 상태)를
# 이미지로 "굽어서" 재현 가능하게 만드는 것이 목적.
#
# 베이스: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#   - 기존 환경과 동일한 torch/CUDA 버전. torch/torchvision/torchaudio 이미 포함.
#   - runtime 태그 사용 (devel 아님): src/adapters/scgpt_adapter.py가
#     use_fast_transformer=False로 표준 nn.MultiheadAttention만 사용하고,
#     scGPT의 flash-attn 체크포인트(Wqkv 가중치)는 코드에서 in_proj_weight로
#     리매핑해 flash-attn 없이 그대로 재사용하므로 flash-attn을 설치하지 않는다.
#     devel 태그(및 이전의 flash-attn 소스 빌드, MAX_JOBS, ninja 단계)는
#     nvcc(CUDA 컴파일러)를 flash-attn 컴파일 목적으로만 썼는데, 이제 그 컴파일
#     자체가 없으므로 devel이 필요 없다. 이 변경으로 이미지 빌드 시간이
#     (flash-attn 소스 컴파일에 걸리던) 10~30분가량 줄어든다.
#     주의: scgpt 패키지 자체는 import 시 flash_attn을 optional로 시도해 보고
#     (scgpt/model/flash_attn_compat.py) 없으면 조용히 폴백하므로 import가
#     깨지지는 않는다 — 향후 속도를 위해 flash-attn 경로를 되살리려면
#     runtime -> devel로 되돌리고 flash-attn 빌드 단계를 다시 추가할 것.
# =========================================================================
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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
#   - build-essential: requirements.txt의 일부 패키지가 이 환경/버전 조합에
#     대한 사전 빌드 wheel이 없을 경우를 대비한 C/C++ 툴체인 (flash-attn
#     전용은 아니었음 — flash-attn 제거 후에도 유지)
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

# -------------------------------------------------------------------------
# 설치 검증 (빌드 단계에서 바로 깨진 환경을 잡아내기 위함)
#   flash_attn은 의도적으로 설치하지 않으므로(위 FROM 설명 참고) 검증 목록에서
#   제외했다 — scgpt import는 flash_attn 없이도 성공해야 정상이다.
# -------------------------------------------------------------------------
RUN python -c "import torch; print('torch', torch.__version__, torch.version.cuda)" \
    && python -c "import scgpt; print('scgpt ok')" \
    && python -c "from scgpt.tokenizer.gene_tokenizer import GeneVocab; print('scgpt tokenizer (torchtext 의존) ok')" \
    && python -c "import scanpy; print('scanpy', scanpy.__version__)" \
    && python -c "import anndata; print('anndata', anndata.__version__)"

# -------------------------------------------------------------------------
# 애플리케이션 코드 복사 (이후 단계에서 작성할 src/)
# -------------------------------------------------------------------------
COPY src/ /workspace/src/

ENTRYPOINT ["python", "/workspace/src/run.py"]
