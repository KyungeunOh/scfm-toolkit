"""type별 precision/recall/F1 + confusion matrix를 별도 CSV로 기록 (교수님 2026-10-08 요청).
<out>.csv 옆에 <out>_pertype.csv, <out>_confusion.csv 를 만든다."""
import csv
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PERTYPE_FIELDS = ["run_id", "scope", "n_per_type", "seed", "epochs", "micro_batch_size", "lr",
                  "celltype", "n_train", "support", "n_pred", "precision", "recall", "f1"]
CONFUSION_FIELDS = ["run_id", "scope", "n_per_type", "seed", "true_type", "pred_type", "n"]


def _append(path, fields, rows):
    path = Path(path)
    new = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def build_rows(meta, y_true, y_pred, labels, train_counts=None):
    """labels: 고정된 전체 type 목록(예: 18종). 반환: (pertype_rows, confusion_rows)."""
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]
    labels = [str(v) for v in labels]
    train_counts = {str(k): v for k, v in (train_counts or {}).items()}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pt_rows = []
    for i, lab in enumerate(labels):
        pt_rows.append({**{k: meta.get(k, "") for k in ("run_id", "scope", "n_per_type", "seed",
                                                           "epochs", "micro_batch_size", "lr")},
                        "celltype": lab, "n_train": train_counts.get(lab, ""),
                        "support": int(s[i]), "n_pred": int(cm[:, i].sum()),
                        "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                        "f1": round(float(f[i]), 4)})
    cf_rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if cm[i, j]:
                cf_rows.append({**{k: meta.get(k, "") for k in ("run_id", "scope", "n_per_type", "seed")},
                                "true_type": a, "pred_type": b, "n": int(cm[i, j])})
    return pt_rows, cf_rows


def write_pertype(csv_path, meta, y_true, y_pred, labels, train_counts=None):
    csv_path = Path(csv_path)
    pt_rows, cf_rows = build_rows(meta, y_true, y_pred, labels, train_counts)
    _append(csv_path.with_name(csv_path.stem + "_pertype.csv"), PERTYPE_FIELDS, pt_rows)
    _append(csv_path.with_name(csv_path.stem + "_confusion.csv"), CONFUSION_FIELDS, cf_rows)
