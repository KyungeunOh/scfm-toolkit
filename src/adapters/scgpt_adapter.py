"""
adapters/scgpt_adapter.py

scGPT용 ModelAdapter 구현체.
로직 자체는 기존 src/run_annotation.py(공식 Tutorial_Annotation.ipynb 이식본)를 그대로 가져왔고,
ModelAdapter 인터페이스에 맞게 클래스 메서드로 재배치했다.
scGPT 관련 세부사항(토큰화 방식, binning, TransformerModel 파라미터 등)은
전부 이 파일 안에만 있고, pipeline/ 쪽에서는 참조하지 않는다.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class ScGPTAdapter(ModelAdapter):
    name = "scgpt"
    required_config_keys = [
        "reference_path", "query_path", "model_dir",
        "celltype_col", "n_bins", "max_seq_len", "batch_size", "epochs",
    ]
    path_config_keys = ["reference_path", "query_path", "model_dir", "finetuned_model_path"]
    #: finetuned_model_path는 선택 항목 (값이 비어있으면 pipeline/config.py가 건너뜀).
    #: 지정하면 run.py가 fine-tuning을 건너뛰고 이 가중치를 바로 불러와 predict한다.

    # ------------------------------------------------------------------
    # vocab (h5ad validation에서 gene overlap 계산용, 가벼운 로드)
    # ------------------------------------------------------------------
    def load_vocab_genes(self, cfg: Dict[str, Any]) -> set:
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        vocab_file = Path(cfg["model_dir"]) / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file)
        return set(vocab.get_stoi().keys())

    # ------------------------------------------------------------------
    # 데이터 로드
    # ------------------------------------------------------------------
    def load_data(self, cfg: dict) -> Tuple[Any, Any, Dict, int]:
        import scanpy as sc

        ref_path = cfg["reference_path"]
        query_path = cfg["query_path"]
        celltype_col = cfg["celltype_col"]
        source_col = cfg.get("source_celltype_col", celltype_col)

        logger.info(f"Reference 로드: {ref_path}")
        adata = sc.read(ref_path)
        logger.info(f"Query 로드: {query_path}")
        adata_test = sc.read(query_path)

        adata.obs[celltype_col] = adata.obs[source_col].astype("category")
        adata_test.obs[celltype_col] = adata_test.obs[source_col].astype("category")

        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"

        if "gene_name" in adata.var.columns:
            adata.var.set_index(adata.var["gene_name"], inplace=True)
        if "gene_name" in adata_test.var.columns:
            adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)

        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels

        celltype_id_labels = adata.obs[celltype_col].astype("category").cat.codes.values
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs[celltype_col].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()

        logger.info(f"  Reference: {(adata.obs['str_batch']=='0').sum()} 세포")
        logger.info(f"  Query:     {(adata.obs['str_batch']=='1').sum()} 세포")
        logger.info(f"  Cell type: {num_types}종 → {list(id2type.values())}")

        self._vocab_dir = cfg["model_dir"]  # load_vocab_full에서 사용
        return adata, adata_test_raw, id2type, num_types

    def load_vocab_full(self, adata, model_dir: str):
        """scGPT는 vocab 로드 후 adata를 vocab 교집합으로 필터링해야 해서
        base 인터페이스의 preprocess 이전에 한 번 더 필요 - scgpt 전용 스텝."""
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        model_dir = Path(model_dir)
        vocab_file = model_dir / "vocab.json"
        config_file = model_dir / "args.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in adata.var["gene_name"]]
        n_before = adata.n_vars
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        logger.info(f"vocab 교집합: {n_before} → {adata.n_vars} 유전자")

        with open(config_file) as f:
            model_configs = json.load(f)

        return adata, vocab, model_configs

    # ------------------------------------------------------------------
    # 전처리
    # ------------------------------------------------------------------
    def preprocess(self, adata, cfg: dict):
        from scgpt.preprocess import Preprocessor

        data_is_raw = cfg["data_is_raw"]
        filter_gene_by_counts = cfg.get("filter_gene_by_counts", False)
        n_bins: int = cfg["n_bins"]

        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=filter_gene_by_counts,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=data_is_raw,
            result_log1p_key="X_log1p",
            subset_hvg=False,
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,
            result_binned_key="X_binned",
        )

        logger.info("Reference 전처리 중...")
        preprocessor(adata[adata.obs["str_batch"] == "0"], batch_key=None)
        logger.info("Query 전처리 중...")
        preprocessor(adata[adata.obs["str_batch"] == "1"], batch_key=None)

        if "X_binned" not in adata.layers:
            ref_mask = adata.obs["str_batch"] == "0"
            test_mask = adata.obs["str_batch"] == "1"
            adata_ref = adata[ref_mask].copy()
            adata_test = adata[test_mask].copy()
            preprocessor(adata_ref, batch_key=None)
            preprocessor(adata_test, batch_key=None)
            n_genes = adata.n_vars
            X_binned = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
            ref_idx = np.where(ref_mask)[0]
            test_idx = np.where(test_mask)[0]
            X_binned[ref_idx] = (
                adata_ref.layers["X_binned"].toarray()
                if issparse(adata_ref.layers["X_binned"]) else adata_ref.layers["X_binned"]
            )
            X_binned[test_idx] = (
                adata_test.layers["X_binned"].toarray()
                if issparse(adata_test.layers["X_binned"]) else adata_test.layers["X_binned"]
            )
            adata.layers["X_binned"] = X_binned

        return adata

    # ------------------------------------------------------------------
    # 입력 준비 (토큰화 + DataLoader)
    # ------------------------------------------------------------------
    def prepare_inputs(self, adata, cfg: dict, vocab=None):
        from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

        max_seq_len = cfg["max_seq_len"]
        batch_size = cfg["batch_size"]
        eval_batch_size = cfg.get("eval_batch_size", batch_size)
        mask_ratio = cfg.get("mask_ratio", 0.0)
        pad_value = -2
        include_zero_gene = cfg.get("include_zero_gene", False)

        adata_ref = adata[adata.obs["str_batch"] == "0"]
        all_counts = (
            adata_ref.layers["X_binned"].A
            if issparse(adata_ref.layers["X_binned"]) else adata_ref.layers["X_binned"]
        )
        genes = adata_ref.var["gene_name"].tolist()
        celltypes_labels = np.array(adata_ref.obs["celltype_id"].tolist())
        batch_ids = np.array(adata_ref.obs["batch_id"].tolist())

        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)

        (train_data, valid_data, train_ct, valid_ct, train_batch, valid_batch) = train_test_split(
            all_counts, celltypes_labels, batch_ids,
            test_size=1.0 - cfg.get("train_ratio", 0.9),
            shuffle=True, random_state=cfg.get("seed", 42),
        )

        def _tokenize(data, ct, batch):
            tok = tokenize_and_pad_batch(
                data, gene_ids, max_len=max_seq_len, vocab=vocab,
                pad_token="<pad>", pad_value=pad_value,
                append_cls=True, include_zero_gene=include_zero_gene,
            )
            masked = random_mask_value(tok["values"], mask_ratio=mask_ratio, mask_value=-1, pad_value=pad_value)
            return {
                "gene_ids": tok["genes"],
                "values": masked,
                "target_values": tok["values"],
                "batch_labels": torch.from_numpy(batch).long(),
                "celltype_labels": torch.from_numpy(ct).long(),
            }

        train_pt = _tokenize(train_data, train_ct, train_batch)
        valid_pt = _tokenize(valid_data, valid_ct, valid_batch)
        logger.info(f"train: {train_pt['gene_ids'].shape[0]}개, valid: {valid_pt['gene_ids'].shape[0]}개")

        train_loader = DataLoader(SeqDataset(train_pt), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(SeqDataset(valid_pt), batch_size=eval_batch_size, shuffle=False)
        return {"train_loader": train_loader, "valid_loader": valid_loader, "gene_ids": gene_ids, "vocab": vocab}

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    def load_model(self, cfg: dict, num_types: int, device, vocab=None, model_configs=None):
        from scgpt.model import TransformerModel

        model_file = Path(cfg["model_dir"]) / "best_model.pt"
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=3,
            n_cls=num_types,
            vocab=vocab,
            dropout=cfg.get("dropout", 0.2),
            pad_token="<pad>",
            pad_value=-2,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=None,
            domain_spec_batchnorm=False,
            input_emb_style="continuous",
            n_input_bins=cfg["n_bins"],
            cell_emb_style="cls",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )

        logger.info(f"가중치 로드: {model_file}")
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            logger.info("  전체 가중치 로드 성공")
        except Exception:
            model_dict = model.state_dict()
            checkpoint_dict = torch.load(model_file, map_location=device)

            # 이 체크포인트는 use_fast_transformer=True(FlashMHA)로 pretrain된 것이라
            # self-attn 가중치가 Wqkv.weight/bias(결합 QKV 프로젝션)로 저장돼 있다.
            # 지금 모델은 flash-attn이 없어 use_fast_transformer=False(표준
            # nn.MultiheadAttention, in_proj_weight/bias)를 쓰지만, scGPT의 FlashMHA는
            # qkv를 'b s (three h d) -> b s three h d' (three가 가장 바깥쪽 축)로
            # reshape하므로 [Q(d_model); K(d_model); V(d_model)] 순서로 이어붙이고
            # 각 블록 내부를 (head, head_dim) 순서로 나누는 PyTorch in_proj_weight와
            # 레이아웃이 완전히 동일하다. 즉 키 이름만 바꾸면 근사가 아니라 수학적으로
            # 동일한 가중치를 flash-attn 설치 없이 그대로 재사용할 수 있다.
            import re
            wqkv_pattern = re.compile(r"(transformer_encoder\.layers\.\d+\.self_attn)\.Wqkv\.(weight|bias)")
            n_remapped = 0
            remapped_dict = {}
            for k, v in checkpoint_dict.items():
                m = wqkv_pattern.match(k)
                if m:
                    k = f"{m.group(1)}.in_proj_{m.group(2)}"
                    n_remapped += 1
                remapped_dict[k] = v
            checkpoint_dict = remapped_dict
            if n_remapped:
                logger.info(f"  Wqkv → in_proj 키 리매핑: {n_remapped}개 "
                            f"(flash-attn 체크포인트를 표준 attention으로 재사용)")

            matched = {}
            shape_mismatch = []   # 이름은 같은데 shape가 다른 것 (진짜 의심 대상)
            missing_in_ckpt = []  # 현재 모델엔 있는데 체크포인트엔 아예 없는 것
            unused_in_ckpt = []   # 체크포인트엔 있는데 현재 모델엔 없는 것

            for k, v in checkpoint_dict.items():
                if k not in model_dict:
                    unused_in_ckpt.append(k)
                elif v.shape != model_dict[k].shape:
                    shape_mismatch.append((k, tuple(v.shape), tuple(model_dict[k].shape)))
                else:
                    matched[k] = v

            for k in model_dict:
                if k not in checkpoint_dict:
                    missing_in_ckpt.append(k)

            model_dict.update(matched)
            model.load_state_dict(model_dict)

            logger.info(f"  부분 로드: {len(matched)}/{len(model_dict)}개 파라미터")
            logger.info(f"  [missing_in_ckpt] 체크포인트에 키 자체가 없음 ({len(missing_in_ckpt)}개, "
                        f"보통 classifier head 등 task-specific 레이어면 정상):")
            for k in missing_in_ckpt:
                logger.info(f"    - {k}  shape={tuple(model_dict[k].shape)}")
            logger.info(f"  [shape_mismatch] 키는 같은데 shape가 다름 ({len(shape_mismatch)}개, "
                        f"backbone 레이어에서 나오면 설정값(config) 불일치를 의심):")
            for k, ckpt_shape, cur_shape in shape_mismatch:
                logger.info(f"    - {k}  ckpt={ckpt_shape}  current={cur_shape}")
            if unused_in_ckpt:
                logger.info(f"  [unused_in_ckpt] 체크포인트에만 있고 현재 모델엔 없는 키 ({len(unused_in_ckpt)}개):")
                for k in unused_in_ckpt:
                    logger.info(f"    - {k}")

        model.to(device)
        return model

    # ------------------------------------------------------------------
    # fine-tune
    # ------------------------------------------------------------------
    def finetune(self, model, prepared_inputs, cfg: dict, device):
        """
        Reference 세트로 classification head(및 backbone)를 fine-tune한다.
        공식 Tutorial_Annotation.ipynb와 동일하게, val_acc가 가장 좋았던 epoch의
        가중치를 별도로 보관했다가 반환한다 (마지막 epoch이 항상 최선이라는 보장이 없으므로).
        run.py는 이 반환값을 predict()에 넘겨서 실제로 best epoch 모델로 예측하게 된다.
        """
        import copy
        from torch.optim import Adam
        from torch.optim.lr_scheduler import StepLR

        train_loader = prepared_inputs["train_loader"]
        valid_loader = prepared_inputs["valid_loader"]
        vocab = prepared_inputs["vocab"]

        epochs = cfg["epochs"]
        lr = cfg.get("lr", 1e-4)
        amp = cfg.get("amp", True)
        accum_steps = cfg.get("grad_accum_steps", 1)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=cfg.get("schedule_ratio", 0.9))
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        best_val_acc = -1.0
        best_epoch = None
        best_state = None

        logger.info(f"Fine-tuning 시작 (grad_accum_steps={accum_steps}, "
                    f"유효 배치 크기={train_loader.batch_size * accum_steps})")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = correct = total = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                gene_ids_b = batch["gene_ids"].to(device)
                values_b = batch["values"].to(device)
                ct_labels = batch["celltype_labels"].to(device)
                padding_mask = gene_ids_b.eq(vocab["<pad>"])

                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(gene_ids_b, values_b, src_key_padding_mask=padding_mask,
                                batch_labels=None, CLS=True, CCE=False, MVC=False, ECS=False)
                    loss = criterion(out["cls_output"], ct_labels) / accum_steps

                scaler.scale(loss).backward()

                is_last_batch = (step + 1 == len(train_loader))
                if (step + 1) % accum_steps == 0 or is_last_batch:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accum_steps
                correct += (out["cls_output"].argmax(1) == ct_labels).sum().item()
                total += len(ct_labels)

            scheduler.step()

            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for batch in valid_loader:
                    gene_ids_b = batch["gene_ids"].to(device)
                    values_b = batch["values"].to(device)
                    ct_labels = batch["celltype_labels"].to(device)
                    padding_mask = gene_ids_b.eq(vocab["<pad>"])
                    with torch.cuda.amp.autocast(enabled=amp):
                        out = model(gene_ids_b, values_b, src_key_padding_mask=padding_mask,
                                    batch_labels=None, CLS=True)
                    val_correct += (out["cls_output"].argmax(1) == ct_labels).sum().item()
                    val_total += len(ct_labels)

            val_acc = val_correct / max(val_total, 1)
            logger.info(
                f"  Epoch {epoch:2d}/{epochs}  "
                f"loss={total_loss/max(len(train_loader),1):.4f}  "
                f"train_acc={correct/max(total,1):.4f}  "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            logger.info(f"best epoch: {best_epoch}/{epochs} (val_acc={best_val_acc:.4f}) 가중치로 예측 진행")
            model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------
    def predict(self, model, adata, prepared_inputs, id2type: dict, cfg: dict, device):
        from scgpt.tokenizer import tokenize_and_pad_batch

        vocab = prepared_inputs["vocab"]
        gene_ids = prepared_inputs["gene_ids"]

        eval_batch_size = cfg.get("eval_batch_size", cfg["batch_size"])
        max_seq_len = cfg["max_seq_len"]
        pad_value = -2
        amp = cfg.get("amp", True)
        include_zero_gene = cfg.get("include_zero_gene", False)

        adata_test = adata[adata.obs["str_batch"] == "1"].copy()
        test_counts = (
            adata_test.layers["X_binned"].A
            if issparse(adata_test.layers["X_binned"]) else adata_test.layers["X_binned"]
        )

        vocab.set_default_index(vocab["<pad>"])
        tok = tokenize_and_pad_batch(
            test_counts, gene_ids, max_len=max_seq_len, vocab=vocab,
            pad_token="<pad>", pad_value=pad_value,
            append_cls=True, include_zero_gene=include_zero_gene,
        )
        test_loader = DataLoader(
            SeqDataset({"gene_ids": tok["genes"], "values": tok["values"]}),
            batch_size=eval_batch_size, shuffle=False,
        )

        model.eval()
        all_preds, all_scores = [], []
        with torch.no_grad():
            for batch in test_loader:
                gene_ids_b = batch["gene_ids"].to(device)
                values_b = batch["values"].to(device)
                padding_mask = gene_ids_b.eq(vocab["<pad>"])
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(gene_ids_b, values_b, src_key_padding_mask=padding_mask,
                                batch_labels=None, CLS=True)
                probs = torch.softmax(out["cls_output"], dim=1)
                all_preds.extend([id2type[i] for i in probs.argmax(1).cpu().numpy()])
                all_scores.extend(probs.max(1).values.cpu().numpy().tolist())

        adata_test.obs["predictions"] = all_preds
        adata_test.obs["pred_score"] = all_scores
        logger.info("예측 완료")
        return adata_test
