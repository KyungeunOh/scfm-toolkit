import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

STEP_DESCRIPTIONS = {
    1: "Reference/Query h5ad를 로드하고 batch/celltype 라벨을 정리합니다.",
    2: "scGPT vocab을 로드하고, 데이터 유전자와의 교집합만 남깁니다.",
    3: "정규화, log1p, binning 등 scGPT 입력 형식으로 전처리합니다.",
    4: "토큰화 후 학습/검증 DataLoader를 구성합니다.",
    5: "사전학습 가중치(best_model.pt)를 로드합니다.",
    6: "Reference로 classification head를 fine-tune합니다.",
    7: "Fine-tune된 모델로 query 세포의 cell type을 예측합니다.",
}


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,  # rich가 진행상황을 보여주므로 일반 로그는 줄임
        format="%(message)s",
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
    """검증 항목을 하나씩 표에 쌓아서, 끝나면 rich 테이블로 한 번에 보여준다."""
    checks = []

    for key in ["reference_path", "query_path", "model_dir", "output_dir", "n_bins"]:
        ok = key in cfg
        checks.append((f"필수 키: {key}", "✅" if ok else "❌", "" if ok else "config.yaml에 없음"))
        if not ok:
            _print_checks(checks)
            raise ValueError(f"config에 필수 키 없음: {key}")

    for label, p in [("reference_path", cfg["reference_path"]), ("query_path", cfg["query_path"])]:
        ok = Path(p).exists()
        checks.append((f"파일 존재: {label}", "✅" if ok else "❌", p))
        if not ok:
            _print_checks(checks)
            raise FileNotFoundError(f"파일 없음: {p}")

    model_dir = Path(cfg["model_dir"])
    for fname in ["vocab.json", "args.json", "best_model.pt"]:
        ok = (model_dir / fname).exists()
        checks.append((f"모델 파일: {fname}", "✅" if ok else "❌", str(model_dir / fname)))
        if not ok:
            _print_checks(checks)
            raise FileNotFoundError(f"모델 파일 없음: {model_dir / fname}")

    _print_checks(checks)


def _print_checks(checks):
    table = Table(title="config.yaml 검증 결과", show_lines=False)
    table.add_column("항목", style="bold")
    table.add_column("결과", justify="center")
    table.add_column("비고", style="dim")
    for name, status, note in checks:
        table.add_row(name, status, note)
    console.print(table)


def _step_banner(step, total, title):
    desc = STEP_DESCRIPTIONS.get(step, "")
    console.print(
        Panel(
            f"[bold]{desc}[/bold]" if desc else "",
            title=f"[bold cyan]Step {step}/{total}: {title}[/bold cyan]",
            border_style="cyan",
        )
    )


def _step_done(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    console.rule("[bold]scFM Toolkit[/bold]")

    cfg = load_config(args.config)
    validate_config(cfg)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_color = "green" if device.type == "cuda" else "yellow"
    console.print(f"디바이스: [{device_color}]{device}[/{device_color}]")
    if device.type == "cuda":
        console.print(f"  GPU: {torch.cuda.get_device_name(0)}")

    sys.path.insert(0, "/workspace/src")
    from run_annotation import (
        load_data, load_vocab, preprocess,
        prepare_dataloaders, load_model,
        finetune, predict, save_results,
    )

    TOTAL = 7

    _step_banner(1, TOTAL, "데이터 로드")
    logging.getLogger("run_annotation").setLevel(logging.INFO)
    adata, adata_test_raw, id2type, num_types = load_data(cfg)
    _step_done(f"reference {sum(adata.obs['str_batch']=='0')}개 + query {sum(adata.obs['str_batch']=='1')}개, cell type {num_types}종")

    _step_banner(2, TOTAL, "vocab 로드 및 유전자 필터링")
    adata, vocab, model_configs = load_vocab(adata, cfg["model_dir"], cfg)
    _step_done(f"vocab 매칭 후 {adata.n_vars}개 유전자 남음")

    _step_banner(3, TOTAL, "전처리")
    adata = preprocess(adata, cfg)
    _step_done("정규화/binning 완료")

    _step_banner(4, TOTAL, "토크나이징 및 DataLoader 준비")
    train_loader, valid_loader, gene_ids = prepare_dataloaders(adata, vocab, cfg)
    _step_done(f"train batch {len(train_loader)}개, valid batch {len(valid_loader)}개")

    _step_banner(5, TOTAL, "모델 로드")
    model = load_model(cfg["model_dir"], vocab, model_configs, num_types, cfg, device)
    _step_done(f"모델 로드 완료 ({device})")

    _step_banner(6, TOTAL, "Fine-tuning")
    finetune(model, train_loader, valid_loader, vocab, cfg, device)
    _step_done("fine-tuning 완료")

    _step_banner(7, TOTAL, "Query 예측")
    adata_result = predict(model, adata, vocab, gene_ids, id2type, cfg, device)
    _step_done("예측 완료")

    console.rule("[bold]결과 저장[/bold]")
    save_results(adata_result, cfg["output_dir"], cfg)

    metrics_path = Path(cfg["output_dir"]) / "metrics.json"
    summary = f"결과 위치: {cfg['output_dir']}"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        acc = m["accuracy"]
        color = "green" if acc >= 0.7 else ("yellow" if acc >= 0.5 else "red")
        summary += f"\n[bold {color}]accuracy = {acc:.2%}[/bold {color}]  ({m['correct']}/{m['total']})"

    console.print(Panel(f"[bold green]완료![/bold green]\n{summary}", border_style="green"))


if __name__ == "__main__":
    main()
