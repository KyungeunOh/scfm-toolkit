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
from pipeline.data_io import DataValidationError, load_h5ad_full, validate_h5ad
from pipeline.integration import evaluate_integration, save_integration_metrics, save_integration_umap
from pipeline.reference_mapping import knn_label_transfer
from pipeline.report import (
    flag_low_confidence,
    save_environment_report,
    save_metrics,
    save_predictions,
    save_resolved_config,
)

console = Console()

# mode별로 단계 구성 자체가 다르다 (embed는 fine-tuning이 없어 훨씬 짧다) - 그래서
# STEP_DESCRIPTIONS를 mode별 dict로 나눴다. run.py가 아는 "mode 목록"은 여기와
# pipeline/config.py의 IMPLEMENTED_MODES뿐이고, 각 mode의 실제 실행 로직은
# run_finetune_predict()/run_embed() 같은 별도 함수로 분리돼 있다.
STEP_DESCRIPTIONS = {
    "finetune_predict": {
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
    },
    "embed": {
        1: "config.yaml을 검증합니다.",
        2: "reference/query h5ad 파일을 검증합니다 (셀 수, label column, vocab 매칭률).",
        3: "Reference/Query h5ad를 로드하고, 사전학습 모델로 각각 임베딩을 추출합니다 "
           "(fine-tuning 없음 - zero-shot).",
        4: "임베딩 공간에서 k-NN 다수결로 reference의 label을 query에 전파합니다.",
        5: "표준 output 구조로 결과를 저장합니다.",
    },
    "integration": {
        1: "config.yaml을 검증합니다.",
        2: "h5ad 입력을 검증합니다 (셀 수, celltype_col/batch_key, vocab 매칭률).",
        3: "highly variable gene을 선택하고, 사전학습 모델로 전체 데이터를 임베딩합니다 "
           "(fine-tuning 없음 - zero-shot).",
        4: "임베딩 공간에서 UMAP을 그리고, scib로 batch 통합 품질 지표를 계산합니다.",
        5: "비교 기준선(HVG+PCA, fine-tuning도 scGPT도 없는 단순 방법)을 같은 방식으로 계산합니다.",
        6: "표준 output 구조로 결과를 저장합니다.",
    },
}


def setup_logging():
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _step_banner(mode, step, title):
    descriptions = STEP_DESCRIPTIONS[mode]
    console.print(Panel(
        f"[bold]{descriptions.get(step, '')}[/bold]",
        title=f"[bold cyan]Step {step}/{len(descriptions)}: {title}[/bold cyan]",
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
        _step_banner("finetune_predict", 8, "Fine-tuning (저장된 가중치 재사용)")
        model = adapter.load_finetuned_model(model, finetuned_model_path, device)
        _step_done(f"재학습 건너뜀 — 기존 fine-tuned 가중치 재사용: {finetuned_model_path}")
        return model

    _step_banner("finetune_predict", 8, "Fine-tuning")
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
    """mode: finetune_predict - pretrained 모델을 reference로 fine-tune한 뒤 query를
    예측한다. main()이 cfg["mode"]로 여기와 run_embed() 중 하나로 분기시킨다."""

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _step_banner("finetune_predict", 3, "데이터 로드")
    logging.getLogger("adapters.scgpt_adapter").setLevel(logging.INFO)
    adata, adata_test_raw, id2type, num_types = adapter.load_data(cfg)
    _step_done(f"reference {sum(adata.obs['str_batch']=='0')}개 + query {sum(adata.obs['str_batch']=='1')}개, cell type {num_types}종")

    _step_banner("finetune_predict", 4, "vocab 로드 및 유전자 필터링")
    adata, vocab, model_configs = adapter.load_vocab_full(adata, cfg["model_dir"])
    _step_done(f"vocab 매칭 후 {adata.n_vars}개 유전자 남음")

    _step_banner("finetune_predict", 5, "전처리")
    adata = adapter.preprocess(adata, cfg)
    _step_done("정규화/binning 완료")

    _step_banner("finetune_predict", 6, "토크나이징 및 DataLoader 준비")
    prepared = adapter.prepare_inputs(adata, cfg, vocab=vocab)
    _step_done(f"train batch {len(prepared['train_loader'])}개, valid batch {len(prepared['valid_loader'])}개")

    _step_banner("finetune_predict", 7, "모델 로드")
    model = adapter.load_model(cfg, num_types, device, vocab=vocab, model_configs=model_configs)
    _step_done(f"모델 로드 완료 ({device})")

    model = _run_finetune_step(adapter, model, prepared, cfg, device, output_dir)

    _step_banner("finetune_predict", 9, "Query 예측")
    adata_result = adapter.predict(model, adata, prepared, id2type, cfg, device)
    _step_done("예측 완료")

    _step_banner("finetune_predict", 10, "결과 저장")
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


def run_embed(cfg: dict, adapter, device) -> None:
    """
    mode: embed - fine-tuning 없이 사전학습 모델의 임베딩만으로 reference mapping을
    수행한다 (scGPT Tutorial_Reference_Mapping.ipynb 방식: reference/query를 각각
    임베딩한 뒤 k-NN 다수결로 reference의 label을 query에 전파).

    finetune_predict와 달리 adapter.embed()가 self-contained으로 동작한다는 전제라
    (vocab 로드/전처리/모델 로딩을 adapter가 내부에서 알아서 함 - base.py의
    embed() docstring 참고) run.py는 여기서도 여전히 model-agnostic하게 남는다:
    reference/query h5ad를 읽고, adapter.embed()를 두 번 호출하고, 그 결과(순수
    numpy 임베딩)를 pipeline/reference_mapping.py에 넘길 뿐이다.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    celltype_col = cfg.get("celltype_col")
    source_col = cfg.get("source_celltype_col", celltype_col)

    _step_banner("embed", 3, "Reference/Query 로드 및 임베딩 추출")
    adata_ref = load_h5ad_full(cfg["reference_path"])
    adata_query = load_h5ad_full(cfg["query_path"])

    if source_col and celltype_col and source_col in adata_ref.obs.columns:
        adata_ref.obs[celltype_col] = adata_ref.obs[source_col]
    if not celltype_col or celltype_col not in adata_ref.obs.columns:
        raise ConfigError(
            f"mode: embed(reference mapping)는 reference h5ad에 celltype_col "
            f"('{celltype_col}')이 있어야 그 label을 query로 전파할 수 있습니다."
        )

    ref_embeddings = adapter.embed(adata_ref, cfg, device)
    query_embeddings = adapter.embed(adata_query, cfg, device)
    _step_done(
        f"reference {adata_ref.n_obs}개, query {adata_query.n_obs}개 임베딩 완료 "
        f"(embed_dim={ref_embeddings.shape[1]})"
    )

    _step_banner("embed", 4, "k-NN reference mapping")
    ref_labels = adata_ref.obs[celltype_col].astype(str).to_numpy()
    k = cfg.get("knn_k", 10)
    pred_labels, pred_scores = knn_label_transfer(ref_embeddings, ref_labels, query_embeddings, k=k)
    _step_done(f"k={k} 다수결 투표로 label 전파 완료")

    adata_query = adata_query.copy()
    adata_query.obs["predictions"] = pred_labels
    adata_query.obs["pred_score"] = pred_scores

    _step_banner("embed", 5, "결과 저장")
    if source_col and celltype_col and source_col in adata_query.obs.columns:
        adata_query.obs[celltype_col] = adata_query.obs[source_col]
    confidence_summary = flag_low_confidence(adata_query, cfg.get("low_confidence_threshold", 0.5))
    save_predictions(adata_query, output_dir, celltype_col)
    metrics = save_metrics(adata_query, output_dir, celltype_col, confidence_summary)
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


def run_integration(cfg: dict, adapter, device) -> None:
    """
    mode: integration - fine-tuning 없이 사전학습 모델의 임베딩만으로 여러 batch가
    섞인 데이터에서 batch effect는 얼마나 제거되고 cell type은 잘 구분되는지 평가한다
    (scGPT tutorials/zero-shot/Tutorial_ZeroShot_Integration.ipynb 방식 - GitHub 원본을
    직접 fetch해서 전처리/embed_data 호출/scib 지표 계산까지 한 줄씩 대조 확인함).

    mode: embed와 마찬가지로 adapter.embed()를 그대로 재사용한다(self-contained라 새
    adapter 코드가 필요 없음). HVG 선택(seurat_v3 - adata.X가 원본 raw count라고
    전제, scikit-misc 필요)과 UMAP/scib 지표 계산은 pipeline/integration.py의 모델
    무관 로직이다.

    scGPT zero-shot 임베딩뿐 아니라 원본 튜토리얼처럼 HVG+PCA 베이스라인도 같은
    방식으로 계산해서 같이 저장한다 - "그냥 믿어라"가 아니라 scGPT 임베딩이 단순
    PCA보다 실제로 나은지 바로 비교할 수 있게 하기 위함.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    celltype_col = cfg.get("celltype_col")
    batch_key = cfg["batch_key"]
    n_hvg = cfg.get("n_hvg", 3000)

    _step_banner("integration", 3, "HVG 선택 및 임베딩 추출")
    import scanpy as sc

    adata = load_h5ad_full(cfg["data_path"])
    if not celltype_col or celltype_col not in adata.obs.columns:
        raise ConfigError(
            f"mode: integration은 data_path h5ad에 celltype_col('{celltype_col}')이 "
            f"있어야 통합 품질(NMI/ARI/ASW)을 평가할 수 있습니다."
        )
    # 정답 label이 없는 셀(NaN 등)은 원본 튜토리얼과 동일하게 제외.
    valid_mask = adata.obs[celltype_col].astype("category").cat.codes.values >= 0
    adata = adata[valid_mask].copy()

    org_adata = adata.copy()  # HVG로 서브셋하기 전 원본 - 아래 HVG+PCA 베이스라인용

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()

    embeddings = adapter.embed(adata, cfg, device)
    adata.obsm["X_scGPT"] = embeddings
    _step_done(
        f"{adata.n_obs}개 세포, HVG {adata.n_vars}개로 scGPT zero-shot 임베딩 완료 "
        f"(embed_dim={embeddings.shape[1]})"
    )

    _step_banner("integration", 4, "UMAP + scib 통합 품질 지표 (scGPT zero-shot)")
    scgpt_umap_path = save_integration_umap(
        adata, batch_key, celltype_col, output_dir / "integration_umap_scgpt.png",
        embed_key="X_scGPT", title_prefix="scGPT zero-shot",
    )
    scgpt_metrics = evaluate_integration(adata, batch_key, celltype_col, embed_key="X_scGPT")
    _step_done(f"scib 지표 계산 완료, UMAP 저장: {scgpt_umap_path.name}")

    _step_banner("integration", 5, "비교 기준선 계산 (HVG+PCA, fine-tuning/scGPT 없음)")
    baseline_adata = org_adata
    sc.pp.highly_variable_genes(baseline_adata, n_top_genes=n_hvg, flavor="seurat_v3")
    baseline_adata = baseline_adata[:, baseline_adata.var["highly_variable"]].copy()
    sc.pp.pca(baseline_adata, n_comps=40)
    baseline_umap_path = save_integration_umap(
        baseline_adata, batch_key, celltype_col, output_dir / "integration_umap_hvg_pca.png",
        embed_key="X_pca", title_prefix="HVG+PCA baseline",
    )
    baseline_metrics = evaluate_integration(baseline_adata, batch_key, celltype_col, embed_key="X_pca")
    _step_done(f"baseline 계산 완료, UMAP 저장: {baseline_umap_path.name}")

    _step_banner("integration", 6, "결과 저장")
    metrics = {"scgpt_zero_shot": scgpt_metrics, "hvg_pca_baseline": baseline_metrics}
    metrics_path = save_integration_metrics(metrics, output_dir)
    save_resolved_config(cfg, output_dir)
    save_environment_report(output_dir)

    summary = f"결과 위치: {output_dir}\n지표: {metrics_path.name}"
    if "avg_bio" in scgpt_metrics and "avg_batch" in scgpt_metrics:
        summary += (
            f"\n[bold]scGPT zero-shot[/bold]: avg_bio={scgpt_metrics['avg_bio']:.3f} "
            f"(높을수록 cell type 분리 잘 됨), avg_batch={scgpt_metrics['avg_batch']:.3f} (높을수록 batch가 잘 섞임)"
        )
    if "avg_bio" in baseline_metrics and "avg_batch" in baseline_metrics:
        summary += (
            f"\n[bold]HVG+PCA baseline[/bold]: avg_bio={baseline_metrics['avg_bio']:.3f}, "
            f"avg_batch={baseline_metrics['avg_batch']:.3f}"
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

        mode = cfg["mode"]
        _step_banner(mode, 1, "config 검증")
        required_keys = adapter.required_config_keys + adapter.extra_required_config_keys(mode)
        path_keys = adapter.path_config_keys + adapter.extra_path_config_keys(mode)
        validate_config(cfg, required_keys, path_keys)
        _step_done(f"model={cfg['model']}, mode={mode}")

        _step_banner(mode, 2, "h5ad 입력 검증")
        vocab_genes = adapter.load_vocab_genes(cfg)
        if mode in ("finetune_predict", "embed"):
            # 이 두 mode는 reference/query 한 쌍을 쓴다.
            validate_h5ad(cfg["reference_path"], "Reference", cfg.get("celltype_col"),
                          cfg.get("source_celltype_col"), vocab_genes)
            validate_h5ad(cfg["query_path"], "Query", cfg.get("celltype_col"),
                          cfg.get("source_celltype_col"), vocab_genes)
        elif mode == "integration":
            # integration은 reference/query 구분이 없는 h5ad 파일 하나(여러 batch가
            # 섞여 있음)를 쓴다 - batch_key는 vocab 매칭과 무관한 obs 컬럼이라
            # validate_h5ad()의 vocab 검증과는 별개로 존재 여부만 확인한다.
            validate_h5ad(cfg["data_path"], "Data", cfg.get("celltype_col"), None, vocab_genes)
            batch_key = cfg.get("batch_key")
            import anndata as ad
            data_obs_columns = ad.read_h5ad(cfg["data_path"], backed="r").obs.columns
            if batch_key not in data_obs_columns:
                raise ConfigError(
                    f"mode: integration은 batch_key('{batch_key}')가 data_path h5ad의 obs 컬럼에 "
                    f"있어야 합니다. 있는 컬럼: {list(data_obs_columns)[:12]}"
                )
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

    if mode == "finetune_predict":
        run_finetune_predict(cfg, adapter, device)
    elif mode == "embed":
        run_embed(cfg, adapter, device)
    elif mode == "integration":
        run_integration(cfg, adapter, device)
    else:
        # pipeline/config.py의 IMPLEMENTED_MODES 검증을 통과했다면 여기 도달할 수 없다 -
        # 도달했다면 IMPLEMENTED_MODES에는 추가했는데 여기 dispatch를 깜빡한 버그.
        raise AssertionError(f"mode='{mode}'가 검증은 통과했지만 실행 분기가 없습니다 (버그).")


if __name__ == "__main__":
    main()
