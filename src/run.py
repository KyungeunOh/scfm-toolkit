import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_config(cfg):
    for key in ["reference_path", "query_path", "model_dir", "output_dir", "n_bins"]:
        if key not in cfg:
            raise ValueError(f"config에 필수 키 없음: {key}")
    for p in [cfg["reference_path"], cfg["query_path"]]:
        if not Path(p).exists():
            raise FileNotFoundError(f"파일 없음: {p}")
    model_dir = Path(cfg["model_dir"])
    for fname in ["vocab.json", "args.json", "best_model.pt"]:
        if not (model_dir / fname).exists():
            raise FileNotFoundError(f"모델 파일 없음: {model_dir / fname}")
    logging.getLogger(__name__).info("config 검증 완료")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"디바이스: {device}")

    sys.path.insert(0, "/workspace/src")
    from run_annotation import (
        load_data, load_vocab, preprocess,
        prepare_dataloaders, load_model,
        finetune, predict, save_results,
    )

    logger.info("=" * 50)
    logger.info("Step 1/7: 데이터 로드")
    adata, adata_test_raw, id2type, num_types = load_data(cfg)

    logger.info("=" * 50)
    logger.info("Step 2/7: vocab 로드 및 유전자 필터링")
    adata, vocab, model_configs = load_vocab(adata, cfg["model_dir"], cfg)

    logger.info("=" * 50)
    logger.info("Step 3/7: 전처리")
    adata = preprocess(adata, cfg)

    logger.info("=" * 50)
    logger.info("Step 4/7: 토크나이징 및 DataLoader 준비")
    train_loader, valid_loader, gene_ids = prepare_dataloaders(adata, vocab, cfg)

    logger.info("=" * 50)
    logger.info("Step 5/7: 모델 로드")
    model = load_model(cfg["model_dir"], vocab, model_configs, num_types, cfg, device)

    logger.info("=" * 50)
    logger.info("Step 6/7: Fine-tuning")
    finetune(model, train_loader, valid_loader, vocab, cfg, device)

    logger.info("=" * 50)
    logger.info("Step 7/7: Query 예측")
    adata_result = predict(model, adata, vocab, gene_ids, id2type, cfg, device)

    logger.info("=" * 50)
    logger.info("결과 저장")
    save_results(adata_result, cfg["output_dir"], cfg)
    logger.info("완료.")

if __name__ == "__main__":
    main()
