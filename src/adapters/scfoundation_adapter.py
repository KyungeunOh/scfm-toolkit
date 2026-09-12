"""
adapters/scfoundation_adapter.py

scFoundation(biomap-research/scFoundation)용 ModelAdapter 구현체 - 2026-09
memory-benchmark 브랜치에서 처음 추가. 교수님이 "입력 gene 수가 늘어날 때
메모리 사용량이 커지는 모델"의 예시로 지목한 모델로, scGPT/Geneformer와
달리 이 프로젝트에 한 번도 붙어본 적이 없다.

================================ 중요: 검증 상태 ================================
아래 구현은 scFoundation 공식 저장소(https://github.com/biomap-research/scFoundation)의
model/finetune_model.py, model/load.py, model/get_embedding.py, model/README.md를
2026-09-12에 직접 읽고 작성했다 (WebFetch로 요약이 아니라 실제 소스를 clone해서
줄 단위로 확인함 - GRN 때(요약 정보만으로 재구현)와 달리 이번엔 원본 코드를 그대로
참고할 수 있었음). 다만:

  - GPU/실제 체크포인트로 단 한 번도 실행해본 적이 없다 (torch/scfoundation 관련
    라이브러리가 이 개발 환경에 없음). 아래 각 메서드의 docstring에 "official 기준"
    이라고 적은 부분은 원본 코드와 대조된 것이고, 그 외(특히 fine-tuning 루프 자체 -
    공식 repo는 forward pass 예시만 주고 학습 루프는 제공하지 않음)는 scgpt_adapter.py의
    구조를 참고해 이 프로젝트 스타일로 새로 작성한 것이다.
  - 체크포인트(models.ckpt)는 PyPI/GitHub가 아니라 별도 SharePoint 링크로만
    배포된다(model/README.md 참고) - config.yaml에 미리 경로를 적어둘 수 없고
    사용자가 직접 받아서 scfoundation_ckpt_path에 지정해야 한다.
  - scFoundation은 pip 패키지가 아니라 저장소 자체를 sys.path에 추가해서
    from load import * 로 쓰는 구조라(공식 README의 사용 예시 그대로), 이 adapter도
    scfoundation_repo_dir(사용자가 clone한 biomap-research/scFoundation 경로)를
    받아서 그 경로의 model/ 폴더를 sys.path에 추가한다.
  - 첫 서버 실행 시 반드시 확인할 것:
      1. requirements-scfoundation.txt의 einops/local_attention이 실제로 이 프로젝트
         Docker 이미지에서 충돌 없이 설치되는지(README "requirements 분리" 절 참고)
      2. load_model_frommmf(ckpt_path, key='gene')의 'gene' key가 실제 다운로드한
         체크포인트 파일 구조와 맞는지 (get_embedding.py는 cell/gene/rde 등 여러
         key를 문맥에 따라 쓰는데, fine-tuning 예시(finetune_model.py)는 'gene'
         하나만 씀 - 우리 목적(분류 fine-tuning)엔 이게 맞다고 판단했지만 실측 확인 필요)
      3. model_config['encoder']['hidden_dim'] 등 config dict의 키 경로가 실제
         로드된 체크포인트에서도 동일한지
      4. gatherData()가 세포마다 "발현된(0이 아닌) 유전자 수"만큼만 실제로 시퀀스
         길이를 만드는지, 그래서 이게 정말 "입력 gene 수 증가 -> 메모리 증가"의
         직접적인 원인이 맞는지(코드상으로는 맞아 보이지만 실측 확인 전까지는 가설)
==================================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .base import ModelAdapter

logger = logging.getLogger(__name__)


def _add_scfoundation_repo_to_path(repo_dir: str) -> None:
    """공식 repo가 pip 패키지가 아니라 'sys.path.append 후 from load import *' 방식이라
    (model/finetune_model.py 1~6행 그대로) 이 함수로 동일하게 흉내낸다."""
    model_dir = str(Path(repo_dir) / "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


class ScFoundationClassifier(nn.Module):
    """
    공식 finetune_model.py의 LinearProbingClassifier를 이 프로젝트의 num_types
    (celltype_col의 실제 클래스 수)에 맞게 일반화한 것. 원본과의 차이:
      - fc1 마지막 레이어 출력이 하드코딩 10 -> num_types 인자로 받음
      - 마지막 2개 transformer block만 풀어주던 것([-2:])을 unfrozen_layers
        인자로 일반화 (README "GPU 메모리 제약 시 일부만 fine-tune" 절의 코드를
        일반화한 것 - 원본 그대로는 아님, 우리가 축으로 스윕하기 위한 확장)
    forward()의 gatherData 기반 sparse gather 로직(0이 아닌 유전자만 모아 시퀀스를
    만드는 부분)은 원본과 동일하게 유지했다 - 이게 "gene 수 증가 -> 메모리 증가"를
    실제로 만드는 핵심 메커니즘이라 원본 그대로 보존하는 게 중요하다고 판단.
    """

    def __init__(self, ckpt_path: str, num_types: int, unfrozen_layers: int = 2, frozen_embeddings: bool = True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.num_types = num_types
        self.unfrozen_layers = unfrozen_layers
        self.frozen_embeddings = frozen_embeddings

    def build(self):
        from load import load_model_frommmf  # scFoundation repo (sys.path에 추가된 상태)

        model, model_config = load_model_frommmf(self.ckpt_path, key="gene")
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        if self.frozen_embeddings:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False

        for _, p in self.encoder.named_parameters():
            p.requires_grad = False
        if self.unfrozen_layers > 0:
            for _, p in self.encoder.transformer_encoder[-self.unfrozen_layers:].named_parameters():
                p.requires_grad = True

        hidden_dim = model_config["encoder"]["hidden_dim"]
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_types),
        )
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)
        self.model_config = model_config
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 19264) - main_gene_selection으로 이미 vocab에 맞게 재배열/0-padding된
        발현량 텐서. gatherData()가 0이 아닌 위치만 골라 실제 transformer 입력
        시퀀스를 만든다(원본 LinearProbingClassifier.forward와 동일한 로직)."""
        from load import gatherData

        value_labels = x > 0
        x_gathered, x_padding = gatherData(x, value_labels, self.model_config["pad_token_id"])
        n_genes = x.shape[1]
        data_gene_ids = torch.arange(n_genes, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config["pad_token_id"])

        emb = self.token_emb(torch.unsqueeze(x_gathered, 2).float(), output_weight=0)
        emb = emb + self.pos_emb(position_gene_ids)

        logits = self.encoder(emb, x_padding)
        logits, _ = torch.max(logits, dim=1)  # (B, hidden_dim) - 원본과 동일한 max-pooling
        logits = self.norm(logits)
        return self.fc1(logits)


class SimpleArrayDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"x": self.x[idx], "celltype_labels": self.y[idx]}


class ScFoundationAdapter(ModelAdapter):
    name = "scfoundation"
    finetuned_model_name = "finetuned_model.pt"

    required_config_keys = [
        "reference_path", "query_path", "celltype_col",
        "scfoundation_repo_dir", "scfoundation_ckpt_path", "scfoundation_gene_list_path",
        "batch_size", "epochs",
    ]
    path_config_keys = [
        "reference_path", "query_path", "scfoundation_repo_dir",
        "scfoundation_ckpt_path", "scfoundation_gene_list_path", "finetuned_model_path",
    ]

    # ------------------------------------------------------------------
    def load_vocab_genes(self, cfg: Dict[str, Any]) -> set:
        """공식 OS_scRNA_gene_index.19264.tsv의 gene_name 컬럼 - scFoundation의
        "vocab"은 scGPT처럼 GeneVocab 객체가 아니라 고정된 19264개 유전자 심볼
        목록이다(model/load.py의 get_genename() 참고, 여기선 경로를 config로
        받으므로 그 로직을 그대로 재현)."""
        import pandas as pd

        df = pd.read_csv(cfg["scfoundation_gene_list_path"], header=0, delimiter="\t")
        return set(df["gene_name"].astype(str))

    # ------------------------------------------------------------------
    def load_data(self, cfg: Dict[str, Any]) -> Tuple[Any, Any, Dict, int]:
        """scgpt_adapter.load_data()와 동일한 패턴(reference+query concatenate,
        str_batch 0/1 태깅) - pipeline/run.py의 finetune_predict 오케스트레이션과
        호환되게 맞췄다. 실제 19264-유전자 재배열은 preprocess()에서 한다(scFoundation
        고유 단계라 scGPT와 분리)."""
        import scanpy as sc

        celltype_col = cfg["celltype_col"]
        source_col = cfg.get("source_celltype_col", celltype_col)

        adata = sc.read(cfg["reference_path"])
        adata_test = sc.read(cfg["query_path"])

        adata.obs[celltype_col] = adata.obs[source_col].astype("category")
        adata_test.obs[celltype_col] = adata_test.obs[source_col].astype("category")
        adata.obs["str_batch"] = "0"
        adata_test.obs["str_batch"] = "1"

        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

        celltype_id_labels = adata.obs[celltype_col].astype("category").cat.codes.values
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs[celltype_col].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels

        logger.info(f"  Reference: {(adata.obs['str_batch']=='0').sum()} 세포, "
                    f"Query: {(adata.obs['str_batch']=='1').sum()} 세포, cell type {num_types}종")
        return adata, adata_test_raw, id2type, num_types

    def load_vocab_full(self, adata, model_dir: str):
        """base.py 인터페이스상 이름은 model_dir이지만, scFoundation은 이 시점에
        무거운 체크포인트를 로드하지 않는다(모델 자체는 load_model()에서 로드) -
        여기서는 19264-유전자 목록만 가볍게 읽어 vocab으로 반환한다. adata 필터링은
        하지 않고 그대로 통과시킨다 - scFoundation은 "vocab과의 교집합만 남기기"가
        아니라 "vocab 전체 길이로 재배열하며 없는 유전자는 0으로 채우기" 방식이라
        (main_gene_selection) 필터링 자체가 다른 연산이고, 실제 재배열은
        preprocess()에서 한다."""
        raise NotImplementedError(
            "ScFoundationAdapter.load_vocab_full()은 run.py의 finetune_predict 경로에서 "
            "직접 쓰이지 않는다 - 대신 preprocess()가 cfg['scfoundation_gene_list_path']를 "
            "다시 읽어 main_gene_selection으로 재배열한다. (설계 노트: base.py 인터페이스가 "
            "model_dir 하나만 받는 시그니처라 이 adapter가 필요로 하는 "
            "scfoundation_gene_list_path/scfoundation_repo_dir을 여기서 받을 수 없어서, "
            "이 메서드 대신 preprocess()/load_model()에서 cfg 전체를 직접 읽는 방식으로 "
            "우회했다 - run.py를 고치지 않기 위한 선택. 정공법은 base.py의 load_vocab_full "
            "시그니처를 cfg 전체를 받도록 넓히는 것인데, 이건 scgpt/geneformer 두 adapter에도 "
            "영향을 주는 변경이라 이번 브랜치 범위 밖으로 남겨둠.)"
        )

    # ------------------------------------------------------------------
    def preprocess(self, adata, cfg: Dict[str, Any]):
        """정규화(옵션) + main_gene_selection으로 19264-유전자 순서에 맞게 재배열.
        official 기준: get_embedding.py의 main_gene_selection()을 그대로 사용
        (adapters/scfoundation_adapter.py 안에서 재구현하지 않고 원본 함수를 그대로
        import) - 없는 유전자는 0으로 채우고, gene_list 순서로 컬럼을 재정렬한다."""
        import pandas as pd
        import scanpy as sc

        from get_embedding import main_gene_selection  # scFoundation repo, sys.path에 이미 추가됨

        if cfg.get("data_is_raw", False):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        # --- 입력 gene 수 축 (2026-09 memory-benchmark 확장, official 코드에는 없음) ---
        # "입력 gene 수를 줄이는 것은 batch size를 줄이는 것과 의미가 다르다"(생물학적
        # 정보 손실 vs 시스템 설정)는 교수님 지적에 대응하기 위해 추가한 축. 원본
        # adata를 top n_hvg_genes개 HVG로 먼저 줄인 뒤 19264-벡터로 재배열하면, 선택
        # 안 된 유전자는 0으로 채워져(main_gene_selection) gatherData가 실제로 모으는
        # "0이 아닌 유전자 수"가 자연스럽게 줄어든다 - 이게 진짜 시퀀스 길이/메모리에
        # 영향을 주는지는 서버에서 실측 확인 필요(모듈 docstring의 검증 상태 참고).
        n_hvg_genes = cfg.get("n_hvg_genes")
        if n_hvg_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_genes, flavor="seurat_v3" if cfg.get("data_is_raw", False) else "cell_ranger")
            adata = adata[:, adata.var["highly_variable"]].copy()
            logger.info(f"n_hvg_genes={n_hvg_genes}로 HVG 서브셋 완료 (재배열 전 {adata.n_vars}개 유전자로 축소)")

        gene_list_df = pd.read_csv(cfg["scfoundation_gene_list_path"], header=0, delimiter="\t")
        gene_list = list(gene_list_df["gene_name"].astype(str))

        gene_col = cfg.get("gene_col")
        gene_names = adata.var[gene_col].astype(str) if gene_col and gene_col in adata.var.columns else adata.var_names.astype(str)

        X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        X_df = pd.DataFrame(X, index=adata.obs_names, columns=gene_names)
        # 중복 유전자 이름이 있으면(드묾) main_gene_selection의 컬럼 select가 깨지므로 명시적으로 방지.
        X_df = X_df.loc[:, ~X_df.columns.duplicated()]

        X_reindexed, to_fill_columns, _var = main_gene_selection(X_df, gene_list)
        n_missing = len(to_fill_columns)
        logger.info(
            f"main_gene_selection: {adata.n_vars}개 유전자 -> {len(gene_list)}개 vocab 기준으로 재배열 "
            f"({n_missing}개는 데이터에 없어 0으로 채움, {len(gene_list) - n_missing}개 실제 매칭)"
        )

        adata.uns["scfoundation_gene_list"] = gene_list
        adata.uns["scfoundation_X_reindexed"] = X_reindexed.values.astype(np.float32)
        return adata

    # ------------------------------------------------------------------
    def prepare_inputs(self, adata, cfg: Dict[str, Any], vocab=None):
        batch_size = cfg["batch_size"]
        eval_batch_size = cfg.get("eval_batch_size", batch_size)

        X_full = adata.uns["scfoundation_X_reindexed"]
        ref_mask = (adata.obs["str_batch"] == "0").to_numpy()
        X_ref = X_full[ref_mask]
        y_ref = adata.obs.loc[ref_mask, "celltype_id"].to_numpy()

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_ref, y_ref, test_size=1.0 - cfg.get("train_ratio", 0.9),
            shuffle=True, random_state=cfg.get("seed", 42),
        )

        train_ds = SimpleArrayDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
        valid_ds = SimpleArrayDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid).long())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False)
        logger.info(f"train: {len(train_ds)}개, valid: {len(valid_ds)}개 (19264-유전자 dense 텐서)")
        return {"train_loader": train_loader, "valid_loader": valid_loader}

    # ------------------------------------------------------------------
    def load_model(self, cfg: Dict[str, Any], num_types: int, device, vocab=None, model_configs=None):
        _add_scfoundation_repo_to_path(cfg["scfoundation_repo_dir"])
        unfrozen_layers = cfg.get("scfoundation_unfrozen_layers", 2)
        model = ScFoundationClassifier(
            ckpt_path=cfg["scfoundation_ckpt_path"], num_types=num_types,
            unfrozen_layers=unfrozen_layers,
        ).build()
        model.to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"모델 로드 완료 - 학습 가능 파라미터: {n_trainable:,}/{n_total:,} "
                    f"(unfrozen_layers={unfrozen_layers})")
        return model

    # ------------------------------------------------------------------
    def finetune(self, model, prepared_inputs, cfg: Dict[str, Any], device):
        """official repo는 학습 루프를 제공하지 않아(forward pass 예시만 있음)
        scgpt_adapter.py의 finetune() 구조(best val_acc epoch 보관, grad_accum,
        precision)를 그대로 이식했다 - scFoundation 고유 로직은 forward()
        (ScFoundationClassifier 안, gatherData 기반)에만 있고 이 학습 루프 자체는
        모델 무관 표준 분류 학습이다."""
        import copy
        from torch.optim import Adam
        from torch.optim.lr_scheduler import StepLR

        train_loader = prepared_inputs["train_loader"]
        valid_loader = prepared_inputs["valid_loader"]

        epochs = cfg["epochs"]
        lr = cfg.get("lr", 1e-4)
        accum_steps = cfg.get("grad_accum_steps", 1)

        precision = cfg.get("precision")
        if precision is None:
            amp, autocast_dtype = cfg.get("amp", True), torch.float16
        elif precision == "fp32":
            amp, autocast_dtype = False, torch.float32
        elif precision == "fp16":
            amp, autocast_dtype = True, torch.float16
        elif precision == "bf16":
            amp, autocast_dtype = True, torch.bfloat16
        else:
            raise ValueError(f"precision='{precision}'은 지원하지 않습니다 (fp32/fp16/bf16 중 하나).")
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and autocast_dtype == torch.float16))

        if cfg.get("activation_checkpointing", False):
            logger.warning(
                "scfoundation_adapter는 activation_checkpointing을 아직 지원하지 않습니다 "
                "(scGPT와 달리 encoder 내부 구조(performer/flash 등 select_model() 선택에 따라 "
                "달라짐)가 확인되지 않아 안전하게 wrapping할 방법을 아직 검증 못함 - README "
                "'남은 작업' 참고). 이 옵션은 무시하고 계속 진행합니다."
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=cfg.get("schedule_ratio", 0.9))

        best_val_acc, best_epoch, best_state = -1.0, None, None
        logger.info(f"Fine-tuning 시작 (grad_accum_steps={accum_steps}, precision={precision or 'amp'})")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = correct = total = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                x = batch["x"].to(device)
                labels = batch["celltype_labels"].to(device)
                with torch.cuda.amp.autocast(enabled=amp, dtype=autocast_dtype):
                    logits = model(x)
                    loss = criterion(logits, labels) / accum_steps
                scaler.scale(loss).backward()

                is_last = (step + 1 == len(train_loader))
                if (step + 1) % accum_steps == 0 or is_last:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accum_steps
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
            scheduler.step()

            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for batch in valid_loader:
                    x = batch["x"].to(device)
                    labels = batch["celltype_labels"].to(device)
                    with torch.cuda.amp.autocast(enabled=amp, dtype=autocast_dtype):
                        logits = model(x)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += len(labels)
            val_acc = val_correct / max(val_total, 1)
            logger.info(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/max(len(train_loader),1):.4f}  "
                        f"train_acc={correct/max(total,1):.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            logger.info(f"best epoch: {best_epoch}/{epochs} (val_acc={best_val_acc:.4f}) 가중치로 예측 진행")
            model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    def save_finetuned_model(self, model, path) -> None:
        torch.save(model.state_dict(), path)

    def load_finetuned_model(self, model, path, device):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model

    # ------------------------------------------------------------------
    def predict(self, model, adata, prepared_inputs, id2type: Dict, cfg: Dict[str, Any], device):
        X_full = adata.uns["scfoundation_X_reindexed"]
        query_mask = (adata.obs["str_batch"] == "1").to_numpy()
        X_query = torch.from_numpy(X_full[query_mask])
        adata_test = adata[query_mask].copy()

        eval_batch_size = cfg.get("eval_batch_size", cfg["batch_size"])
        loader = DataLoader(X_query, batch_size=eval_batch_size, shuffle=False)

        model.eval()
        all_preds, all_scores = [], []
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_preds.extend([id2type[i] for i in probs.argmax(1).cpu().numpy()])
                all_scores.extend(probs.max(1).values.cpu().numpy().tolist())

        adata_test.obs["predictions"] = all_preds
        adata_test.obs["pred_score"] = all_scores
        return adata_test
