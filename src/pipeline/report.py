"""
pipeline/report.py
실행 결과를 표준화된 output 구조로 저장.

기존에는 predictions.h5ad + metrics.json(accuracy만) 두 개였는데,
여기서는 다음을 표준 output으로 정의한다:

  output_dir/
    predictions.csv          - 셀 단위 예측 결과 (경량, 바로 열어볼 수 있음)
    predictions.h5ad         - 원본 데이터 + 예측 결과가 담긴 h5ad
    metrics.json             - 전체 accuracy, 예측 신뢰도 요약 등
    per_class_metrics.csv    - label이 있을 경우 클래스별 precision/recall/f1
    confusion_matrix.png     - label이 있을 경우 confusion matrix 시각화
    finetuned_model.pt       - fine-tune된 모델 가중치 (run.py가 저장, 재사용 가능)
    resolved_config.yaml     - 기본값까지 채워서 실제로 사용된 config 전체
    environment.json         - 실행 환경 기록 (라이브러리 버전, GPU, git commit)
"""

import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def flag_low_confidence(adata_result, threshold: Optional[float]) -> Optional[Dict]:
    """
    예측 신뢰도(softmax 최대 확률, pred_score)가 threshold 미만인 셀을
    obs['low_confidence']로 표시하고 요약 통계를 반환한다.

    label(정답 celltype)이 없는 순수 예측 상황에서도 동작한다 — accuracy와 달리
    신뢰도는 예측 자체(pred_score)만 있으면 계산할 수 있으므로, 비전문가가 정답
    라벨 없이 새 데이터를 예측할 때도 "이 예측을 얼마나 믿을 수 있는지" 신호를 준다.

    threshold가 None이면 기능을 비활성화하고 None을 반환한다 (config.yaml에서
    low_confidence_threshold: null 로 명시적으로 끌 수 있음).
    """
    if threshold is None or "pred_score" not in adata_result.obs.columns:
        return None

    scores = adata_result.obs["pred_score"].astype(float)
    low_mask = scores < threshold
    adata_result.obs["low_confidence"] = low_mask.values

    total = int(len(low_mask))
    count = int(low_mask.sum())
    ratio = count / total if total else 0.0

    logger.info(
        f"신뢰도 < {threshold:.0%}인 예측: {count}/{total} ({ratio:.1%})"
    )
    return {
        "low_confidence_threshold": float(threshold),
        "low_confidence_count": count,
        "low_confidence_total": total,
        "low_confidence_ratio": ratio,
    }


def save_predictions(adata_result, output_dir: Path, celltype_col: Optional[str]) -> Path:
    """predictions.h5ad + predictions.csv 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = output_dir / "predictions.h5ad"
    adata_result.write_h5ad(h5ad_path)

    cols = ["predictions", "pred_score"]
    if "low_confidence" in adata_result.obs.columns:
        cols.append("low_confidence")
    if celltype_col and celltype_col in adata_result.obs.columns:
        cols = [celltype_col] + cols
    csv_path = output_dir / "predictions.csv"
    adata_result.obs[[c for c in cols if c in adata_result.obs.columns]].to_csv(csv_path)

    logger.info(f"예측 결과 저장: {h5ad_path.name}, {csv_path.name}")
    return h5ad_path


def save_metrics(
    adata_result,
    output_dir: Path,
    celltype_col: Optional[str],
    confidence_summary: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    metrics.json에 다음을 담는다:
      - confidence_summary가 있으면 low_confidence_* 통계 (label 유무와 무관하게 항상 포함)
      - label(celltype_col)이 있으면 추가로:
          - accuracy 등 요약
          - per_class_metrics.csv (precision/recall/f1 per class)
          - confusion_matrix.png
    label도 없고 confidence_summary도 없으면 metrics.json 자체를 생성하지 않고
    None을 반환한다 (아무것도 계산할 게 없는 경우).
    """
    metrics: Dict = {}
    if confidence_summary:
        metrics.update(confidence_summary)

    has_labels = bool(celltype_col) and celltype_col in adata_result.obs.columns and \
        "predictions" in adata_result.obs.columns

    if has_labels:
        from sklearn.metrics import classification_report, confusion_matrix

        y_true = adata_result.obs[celltype_col].astype(str)
        y_pred = adata_result.obs["predictions"].astype(str)

        correct = int((y_true == y_pred).sum())
        total = int(len(y_true))
        accuracy = correct / total if total else 0.0

        metrics["accuracy"] = accuracy
        metrics["correct"] = correct
        metrics["total"] = total

        # per-class
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        import csv as csv_module
        per_class_path = output_dir / "per_class_metrics.csv"
        with open(per_class_path, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow(["class", "precision", "recall", "f1-score", "support"])
            for cls, vals in report.items():
                if cls in ("accuracy", "macro avg", "weighted avg"):
                    continue
                writer.writerow([cls, vals["precision"], vals["recall"], vals["f1-score"], vals["support"]])
            for cls in ("macro avg", "weighted avg"):
                vals = report[cls]
                writer.writerow([cls, vals["precision"], vals["recall"], vals["f1-score"], vals["support"]])

        # confusion matrix 이미지
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = sorted(set(y_true) | set(y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.5)))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (accuracy={accuracy:.2%})")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm[i, j] > 0:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"confusion matrix 이미지 생성 실패 (metrics.json/csv는 정상 저장됨): {e}")

        logger.info(f"metrics 저장: accuracy={accuracy:.2%} ({correct}/{total})")
    else:
        logger.info("label column이 없어 accuracy 계산을 건너뜁니다 (예측 전용 실행).")

    if not metrics:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def save_resolved_config(cfg: Dict, output_dir: Path) -> Path:
    """실제로 사용된 config 전체(기본값 포함)를 저장 - 재현성용"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def save_environment_report(output_dir: Path) -> Path:
    """실행 환경 기록 - 나중에 결과 재현/디버깅할 때 필요"""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    for pkg in ["torch", "scgpt", "scanpy", "anndata", "scikit-learn"]:
        env[f"{pkg}_version"] = _get_package_version(pkg)

    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        env["cuda_available"] = None

    env["git_commit"] = _get_git_commit()

    path = output_dir / "environment.json"
    with open(path, "w") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    return path


def _get_package_version(pkg_name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None


def _get_git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
