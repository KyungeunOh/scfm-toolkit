"""
pipeline/report.py가 만드는 표준 output 구조를 실제로 생성해서 확인/캡처용으로 남긴다.
scGPT 예측 대신 label에 노이즈를 섞은 mock 예측을 사용 (구조 검증이 목적).
"""
import sys
from pathlib import Path

import numpy as np
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline.report import save_predictions, save_metrics, save_resolved_config, save_environment_report

FIXTURES = Path(__file__).parent / "fixtures"
OUT_DIR = Path(__file__).parent / "demo_outputs"

rng = np.random.default_rng(0)
query = ad.read_h5ad(FIXTURES / "query.h5ad")

# mock 예측: 70% 확률로 정답, 30%는 랜덤 오답 (confusion matrix가 의미 있게 보이도록)
celltypes = sorted(query.obs["celltype"].unique().tolist())
preds = []
for true_label in query.obs["celltype"]:
    if rng.random() < 0.7:
        preds.append(true_label)
    else:
        preds.append(rng.choice([c for c in celltypes if c != true_label]))
query.obs["predictions"] = preds
query.obs["pred_score"] = rng.uniform(0.5, 0.99, size=len(query)).round(3)

cfg_used = {
    "model": "scgpt", "mode": "finetune_predict",
    "reference_path": str(FIXTURES / "reference.h5ad"),
    "query_path": str(FIXTURES / "query.h5ad"),
    "model_dir": str(FIXTURES / "model"),
    "output_dir": str(OUT_DIR),
    "celltype_col": "celltype", "n_bins": 51, "max_seq_len": 3001,
    "batch_size": 8, "epochs": 10, "grad_accum_steps": 4,
    "seed": 42,
}

save_predictions(query, OUT_DIR, celltype_col="celltype")
metrics = save_metrics(query, OUT_DIR, celltype_col="celltype")
save_resolved_config(cfg_used, OUT_DIR)
save_environment_report(OUT_DIR)

print("생성된 output:", metrics)
for f in sorted(OUT_DIR.iterdir()):
    print(" -", f.name)
