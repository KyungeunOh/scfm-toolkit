"""
run.py
scFM Toolkit의 얇은 오케스트레이터.

이 파일은 "무엇을 언제 실행할지"만 알고, "어떻게 실행할지"는 모른다.
- config validation, h5ad validation, output 저장  -> pipeline/
- scGPT(또는 향후 Geneformer 등) 실제 실행 로직     -> adapters/

즉 여기서 scgpt import는 한 줄도 없다. 새 모델을 추가해도 이 파일은 안 바뀐다.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))

from adapters import get_adapter
from pipeline.config import ConfigError, load_config, validate_config
from pipeline.data_io import DataValidationError, validate_h5ad
from pipeline.report import (
    flag_low_confidence,
    save_environment_report,
    save_metrics,
    save_predictions,
    save_resolved_config,
)

console = Console()

STEP_DESCRIPTIONS = {
    1: "config.yaml을 검증합니다.",
    2: "reference/query h5ad 파일을 검증합니다 (셀 수, label column, vocab 매칭률).",
    3: "Reference/Query h5ad를 로드하고 batch/celltype 라벨을 정리합니다.",
    4: "모델 vocab을 로드하고, 데이터 유전자와의 교집합만 남깁니다.",
    5: "정규화, log1p, binning 등 모델 입력 형식으로 전처리합니다.",
    6: "토큰화 후 학습/검증 DataLoader를 구성합니다.",
    7: "사전학습 가중치를 로드합니다.",
    8: "Reference로 classification head를 fine-tune합니다 "
       "(finetuned_model_path가 지정되면 재학습 없이 그 가중치를 재사용합니다).",
    9: "Fine-tune된 모델로 query 세포의 cell type을 예측합니다.",
    10: "표준 output 구조로 결과를 저장합니다.",
}
TOTAL_STEPS = len(STEP_DESCRIPTIONS)


def setup_logging():
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _step_banner(step, title):
    console.print(Panel(
        f"[bold]{STEP_DESCRIPTIONS.get(step, '')}[/bold]",
        title=f"[bold cyan]Step {step}/{TOTAL_STEPS}: {title}[/bold cyan]",
        border_style="cyan",
    ))


def _step_done(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def _run_finetune_step(adapter, model, prepared, cfg, device, output_dir):
    """
    Step 8: fine-tune 실행부.

    - cfg에 finetuned_model_path가 지정돼 있고 그 경로가 존재하면: 재학습을 완전히
      건너뛰고 저장된 가중치를 그대로 불러온다 (매 predict마다 처음부터 fine-tune해야
      했던 문제를 해결하기 위한 재사용 경로).
    - 아니면: adapter.finetune()으로 새로 학습하고, 결과를
      output_dir/<adapter.finetuned_model_name>에 저장해 다음 실행에서 재사용할 수
      있게 한다.

    저장/불러오기 방식(단일 파일 state_dict인지, 체크포인트 디렉터리인지 등)은
    adapter.save_finetuned_model()/load_finetuned_model()이 전적으로 결정한다 —
    이 함수는 그 형태를 몰라도 되므로 scgpt를 비롯한 모델별 라이브러리를 전혀
    import하지 않는다 (run.py의 설계 원칙 유지).
    """
    finetuned_model_path = cfg.get("finetuned_model_path")

    if finetuned_model_path:
        _step_banner(8, "Fine-tuning (저장된 가중치 재사용)")
        model = adapter.load_finetuned_model(model, finetuned_model_path, device)
        _step_done(f"재학습 건너뜀 — 기존 fine-tuned 가중치 재사용: {finetuned_model_path}")
        return model

    _step_banner(8, "Fine-tuning")
    model = adapter.finetune(model, prepared, cfg, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    finetuned_path = output_dir / adapter.finetuned_model_name
    adapter.save_finetuned_model(model, finetuned_path)
    _step_done(
        f"fine-tuning 완료 (best validation epoch 가중치 적용), 저장: {finetuned_path.name} "
        f"— 다음 실행에서 finetuned_model_path: {finetuned_path} 로 재사용 가능"
    )
    return model


def run_finetune_predict(cfg: dict, adapter, device) -> None:
    """지금 유일하게 구현된 mode. embed/train_head는 pipeline/config.py에서 이미
    '아직 구현 안 됨'으로 막혀 있으므로 여기 도달하는 시점엔 항상 이 mode다."""

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _step_banner(3, "데이터 로드")
    logging.getLogger("adapters.scgpt_adapter").setLevel(logging.INFO)
    adata, adata_test_raw, id2type, num_types = adapter.load_data(cfg)
    _step_done(f"reference {sum(adata.obs['str_batch']=='0')}개 + query {sum(adata.obs['str_batch']=='1')}개, cell type {num_types}종")

    _step_banner(4, "vocab 로드 및 유전자 필터링")
    adata, vocab, model_configs = adapter.load_vocab_full(adata, cfg["model_dir"])
    _step_done(f"vocab 매칭 후 {adata.n_vars}개 유전자 남음")

    _step_banner(5, "전처리")
    adata = adapter.preprocess(adata, cfg)
    _step_done("정규화/binning 완료")

    _step_banner(6, "토크나이징 및 DataLoader 준비")
    prepared = adapter.prepare_inputs(adata, cfg, vocab=vocab)
    _step_done(f"train batch {len(prepared['train_loader'])}개, valid batch {len(prepared['valid_loader'])}개")

    _step_banner(7, "모델 로드")
    model = adapter.load_model(cfg, num_types, device, vocab=vocab, model_configs=model_configs)
    _step_done(f"모델 로드 완료 ({device})")

    model = _run_finetune_step(adapter, model, prepared, cfg, device, output_dir)

    _step_banner(9, "Query 예측")
    adata_result = adapter.predict(model, adata, prepared, id2type, cfg, device)
    _step_done("예측 완료")

    _step_banner(10, "결과 저장")
    celltype_col = cfg.get("celltype_col")
    confidence_summary = flag_low_confidence(adata_result, cfg.get("low_confidence_threshold", 0.5))
    save_predictions(adata_result, output_dir, celltype_col)
    metrics = save_metrics(adata_result, output_dir, celltype_col, confidence_summary)
    save_resolved_config(cfg, output_dir)
    save_environment_report(output_dir)

    summary = f"결과 위치: {output_dir}"
    if metrics and "accuracy" in metrics:
        acc = metrics["accuracy"]
        color = "green" if acc >= 0.7 else ("yellow" if acc >= 0.5 else "red")
        summary += f"\n[bold {color}]accuracy = {acc:.2%}[/bold {color}]  ({metrics['correct']}/{metrics['total']})"
    if metrics and "low_confidence_ratio" in metrics:
        lc_ratio = metrics["low_confidence_ratio"]
        lc_color = "red" if lc_ratio >= 0.3 else ("yellow" if lc_ratio >= 0.1 else "green")
        summary += (
            f"\n[bold {lc_color}]신뢰도 낮은 예측(<{metrics['low_confidence_threshold']:.0%}) = {lc_ratio:.1%}[/bold {lc_color}]"
            f"  ({metrics['low_confidence_count']}/{metrics['low_confidence_total']}, predictions.csv의 low_confidence 컬럼 참고)"
        )
    console.print(Panel(f"[bold green]완료![/bold green]\n{summary}", border_style="green"))


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    console.rule("[bold]scFM Toolkit[/bold]")

    try:
        cfg = load_config(args.config)
        adapter = get_adapter(cfg["model"])

        _step_banner(1, "config 검증")
        validate_config(cfg, adapter.required_config_keys, adapter.path_config_keys)
        _step_done(f"model={cfg['model']}, mode={cfg['mode']}")

        _step_banner(2, "h5ad 입력 검증")
        vocab_genes = adapter.load_vocab_genes(cfg)
        validate_h5ad(cfg["reference_path"], "Reference", cfg.get("celltype_col"),
                      cfg.get("source_celltype_col"), vocab_genes)
        validate_h5ad(cfg["query_path"], "Query", cfg.get("celltype_col"),
                      cfg.get("source_celltype_col"), vocab_genes)
        _step_done("입력 검증 통과")

    except (ConfigError, DataValidationError) as e:
        console.print(Panel(f"[bold red]검증 실패[/bold red]\n{e}", border_style="red", title="실행 중단"))
        sys.exit(1)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_color = "green" if device.type == "cuda" else "yellow"
    console.print(f"디바이스: [{device_color}]{device}[/{device_color}]")
    if device.type == "cuda":
        console.print(f"  GPU: {torch.cuda.get_device_name(0)}")

    run_finetune_predict(cfg, adapter, device)


if __name__ == "__main__":
    main()
