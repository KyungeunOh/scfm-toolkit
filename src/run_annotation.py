"""
run_annotation.py
공식 Tutorial_Annotation.ipynb 코드를 함수 단위로 추출.
모든 경로/파라미터는 config dict로만 주입됨.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class SeqDataset(Dataset):
    """튜토리얼 코드 그대로 - 외부 클래스 의존 없음."""
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def load_data(cfg: dict):
    ref_path     = cfg["reference_path"]
    query_path   = cfg["query_path"]
    celltype_col = cfg["celltype_col"]
    source_col   = cfg.get("source_celltype_col", celltype_col)

    logger.info(f"Reference 로드: {ref_path}")
    adata = sc.read(ref_path)
    logger.info(f"Query 로드: {query_path}")
    adata_test = sc.read(query_path)

    adata.obs[celltype_col]      = adata.obs[source_col].astype("category")
    adata_test.obs[celltype_col] = adata_test.obs[source_col].astype("category")

    adata.obs["batch_id"]      = adata.obs["str_batch"]      = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"

    if "gene_name" in adata.var.columns:
        adata.var.set_index(adata.var["gene_name"], inplace=True)
    if "gene_name" in adata_test.var.columns:
        adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)

    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")

    batch_id_labels    = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels

    celltype_id_labels = adata.obs[celltype_col].astype("category").cat.codes.values
    num_types          = len(np.unique(celltype_id_labels))
    id2type            = dict(enumerate(adata.obs[celltype_col].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"]   = adata.var.index.tolist()

    logger.info(f"  Reference: {(adata.obs['str_batch']=='0').sum()} 세포")
    logger.info(f"  Query:     {(adata.obs['str_batch']=='1').sum()} 세포")
    logger.info(f"  Cell type: {num_types}종 → {list(id2type.values())}")
    return adata, adata_test_raw, id2type, num_types


def load_vocab(adata, model_dir: str, cfg: dict):
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    model_dir   = Path(model_dir)
    vocab_file  = model_dir / "vocab.json"
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

def preprocess(adata, cfg: dict):
    from scgpt.preprocess import Preprocessor

    data_is_raw           = cfg["data_is_raw"]
    filter_gene_by_counts = cfg.get("filter_gene_by_counts", False)
    n_bins: int           = cfg["n_bins"]

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

    # view가 아닌 원본 adata에 X_binned가 있는지 확인
    # 없으면 전처리 결과를 직접 원본에 주입
    if "X_binned" not in adata.layers:
        import numpy as np
        ref_mask  = adata.obs["str_batch"] == "0"
        test_mask = adata.obs["str_batch"] == "1"
        adata_ref  = adata[ref_mask].copy()
        adata_test = adata[test_mask].copy()
        preprocessor(adata_ref,  batch_key=None)
        preprocessor(adata_test, batch_key=None)
        from scipy.sparse import issparse
        import numpy as np
        n_genes = adata.n_vars
        X_binned = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
        ref_idx  = np.where(ref_mask)[0]
        test_idx = np.where(test_mask)[0]
        X_binned[ref_idx]  = adata_ref.layers["X_binned"].toarray() if issparse(adata_ref.layers["X_binned"]) else adata_ref.layers["X_binned"]
        X_binned[test_idx] = adata_test.layers["X_binned"].toarray() if issparse(adata_test.layers["X_binned"]) else adata_test.layers["X_binned"]
        adata.layers["X_binned"] = X_binned

    return adata

def prepare_dataloaders(adata, vocab, cfg: dict):
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

    max_seq_len       = cfg["max_seq_len"]
    batch_size        = cfg["batch_size"]
    eval_batch_size   = cfg.get("eval_batch_size", batch_size)
    mask_ratio        = cfg.get("mask_ratio", 0.0)
    pad_value         = -2
    include_zero_gene = cfg.get("include_zero_gene", False)

    adata_ref = adata[adata.obs["str_batch"] == "0"]
    all_counts = (
        adata_ref.layers["X_binned"].A
        if issparse(adata_ref.layers["X_binned"])
        else adata_ref.layers["X_binned"]
    )
    genes            = adata_ref.var["gene_name"].tolist()
    celltypes_labels = np.array(adata_ref.obs["celltype_id"].tolist())
    batch_ids        = np.array(adata_ref.obs["batch_id"].tolist())

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    (train_data, valid_data,
     train_ct, valid_ct,
     train_batch, valid_batch) = train_test_split(
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
        masked = random_mask_value(
            tok["values"], mask_ratio=mask_ratio,
            mask_value=-1, pad_value=pad_value,
        )
        return {
            "gene_ids":        tok["genes"],
            "values":          masked,
            "target_values":   tok["values"],
            "batch_labels":    torch.from_numpy(batch).long(),
            "celltype_labels": torch.from_numpy(ct).long(),
        }

    train_pt = _tokenize(train_data, train_ct, train_batch)
    valid_pt = _tokenize(valid_data, valid_ct, valid_batch)

    logger.info(f"train: {train_pt['gene_ids'].shape[0]}개, valid: {valid_pt['gene_ids'].shape[0]}개")

    train_loader = DataLoader(SeqDataset(train_pt), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(SeqDataset(valid_pt), batch_size=eval_batch_size, shuffle=False)
    return train_loader, valid_loader, gene_ids


def load_model(model_dir: str, vocab, model_configs: dict,
               num_types: int, cfg: dict, device: torch.device):
    from scgpt.model import TransformerModel

    model_file = Path(model_dir) / "best_model.pt"

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
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"  부분 로드: {len(pretrained_dict)}개 파라미터")

    model.to(device)
    return model

def finetune(model, train_loader, valid_loader, vocab, cfg: dict, device: torch.device):
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR

    epochs      = cfg["epochs"]
    lr          = cfg.get("lr", 1e-4)
    amp         = cfg.get("amp", True)
    accum_steps = cfg.get("grad_accum_steps", 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.get("schedule_ratio", 0.9))
    scaler    = torch.cuda.amp.GradScaler(enabled=amp)

    logger.info(f"Fine-tuning 시작 (grad_accum_steps={accum_steps}, "
                f"유효 배치 크기={train_loader.batch_size * accum_steps})")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = total = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            gene_ids_b  = batch["gene_ids"].to(device)
            values_b    = batch["values"].to(device)
            ct_labels   = batch["celltype_labels"].to(device)
            padding_mask = gene_ids_b.eq(vocab["<pad>"])

            with torch.cuda.amp.autocast(enabled=amp):
                out  = model(gene_ids_b, values_b,
                             src_key_padding_mask=padding_mask,
                             batch_labels=None, CLS=True,
                             CCE=False, MVC=False, ECS=False)
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
            correct    += (out["cls_output"].argmax(1) == ct_labels).sum().item()
            total      += len(ct_labels)

        scheduler.step()

        # 검증
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in valid_loader:
                gene_ids_b   = batch["gene_ids"].to(device)
                values_b     = batch["values"].to(device)
                ct_labels    = batch["celltype_labels"].to(device)
                padding_mask = gene_ids_b.eq(vocab["<pad>"])
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(gene_ids_b, values_b,
                                src_key_padding_mask=padding_mask,
                                batch_labels=None, CLS=True)
                val_correct += (out["cls_output"].argmax(1) == ct_labels).sum().item()
                val_total   += len(ct_labels)

        logger.info(
            f"  Epoch {epoch:2d}/{epochs}  "
            f"loss={total_loss/max(len(train_loader),1):.4f}  "
            f"train_acc={correct/max(total,1):.4f}  "
            f"val_acc={val_correct/max(val_total,1):.4f}"
        )

def predict(model, adata, vocab, gene_ids: np.ndarray,
            id2type: dict, cfg: dict, device: torch.device):
    from scgpt.tokenizer import tokenize_and_pad_batch

    eval_batch_size   = cfg.get("eval_batch_size", cfg["batch_size"])
    max_seq_len       = cfg["max_seq_len"]
    pad_value         = -2
    amp               = cfg.get("amp", True)
    include_zero_gene = cfg.get("include_zero_gene", False)

    adata_test = adata[adata.obs["str_batch"] == "1"].copy()
    test_counts = (
        adata_test.layers["X_binned"].A
        if issparse(adata_test.layers["X_binned"])
        else adata_test.layers["X_binned"]
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
            gene_ids_b   = batch["gene_ids"].to(device)
            values_b     = batch["values"].to(device)
            padding_mask = gene_ids_b.eq(vocab["<pad>"])
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(gene_ids_b, values_b,
                            src_key_padding_mask=padding_mask,
                            batch_labels=None, CLS=True)
            probs = torch.softmax(out["cls_output"], dim=1)
            all_preds.extend([id2type[i] for i in probs.argmax(1).cpu().numpy()])
            all_scores.extend(probs.max(1).values.cpu().numpy().tolist())

    adata_test.obs["predictions"] = all_preds
    adata_test.obs["pred_score"]  = all_scores
    logger.info("예측 완료")
    return adata_test


def save_results(adata_test, output_dir: str, cfg: dict):
    os.makedirs(output_dir, exist_ok=True)
    out_h5ad = Path(output_dir) / "predictions.h5ad"
    adata_test.write_h5ad(out_h5ad)
    logger.info(f"결과 저장: {out_h5ad}")

    celltype_col = cfg.get("celltype_col")
    if celltype_col and celltype_col in adata_test.obs and "predictions" in adata_test.obs:
        correct  = (adata_test.obs[celltype_col].astype(str) == adata_test.obs["predictions"].astype(str)).sum()
        total    = len(adata_test)
        accuracy = correct / total
        metrics  = {"accuracy": float(accuracy), "correct": int(correct), "total": int(total)}
        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"정확도: {accuracy:.4f} ({correct}/{total})")
