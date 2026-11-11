#!/usr/bin/env python3
"""희소 클래스 요약: <out>_pertype.csv 에서 run 별 macro F1 을
(전체 / 학습 세포 수 n_train >= T 인 클래스 / n_train < T 인 클래스) 로 나눠 보여 준다.

사용:
  python3 benchmark/tools/rare_class_summary.py benchmark_results/scfoundation_sweep_lr_all_s45.csv
  python3 benchmark/tools/rare_class_summary.py <메인 CSV> --min-train 10 --out summary.csv

- 입력은 메인 CSV 경로다. 같은 이름의 _pertype.csv 를 자동으로 읽는다. 메인 CSV 의 CSV 스키마는 바꾸지 않는다.
- 전체 macro F1 이 메인 CSV 의 macro_f1 과 어긋나면 diff_all_vs_csv 열에 나타난다
  (두 값의 정의/예측이 다르면 0 이 아니다 - 0 이 아니면 이 표를 그대로 믿지 말고 원인을 먼저 확인할 것).
- n>=T 분리는 사후(post hoc) 분할이다. 기본 지표를 바꾸는 것이 아니라 보조 지표(탐색적)로 쓴다.
"""
import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x):
    v = _to_float(x)
    return 0 if v is None else int(v)  # n_train 이 비어 있으면(학습에 없던 클래스) 0 으로 본다


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _fmt(v):
    return "" if v is None else round(v, 4) + 0.0  # -0.0 방지


def summarize(main_csv, min_train):
    main_csv = Path(main_csv)
    pt_path = main_csv.with_name(main_csv.stem + "_pertype.csv")
    if not pt_path.exists():
        sys.exit(f"{pt_path} 가 없습니다. (pertype 기록이 없는 실행이면 이 도구로 요약할 수 없습니다)")
    runs = OrderedDict()
    with open(pt_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            runs.setdefault(r["run_id"], []).append(r)
    main = {}
    if main_csv.exists():
        with open(main_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                main[r["run_id"]] = r

    out = []
    for run_id, rows in runs.items():
        f1_all = [_to_float(r["f1"]) for r in rows]
        if any(v is None for v in f1_all):
            sys.exit(f"run {run_id}: f1 값이 비어 있는 행이 있습니다.")
        common = [v for v, r in zip(f1_all, rows) if _to_int(r["n_train"]) >= min_train]
        rare_rows = [(r["celltype"], _to_int(r["n_train"]), _to_float(r["f1"]))
                     for r in rows if _to_int(r["n_train"]) < min_train]
        rare = [x[2] for x in rare_rows]
        macro_all = _mean(f1_all)
        csv_f1 = _to_float(main.get(run_id, {}).get("macro_f1"))
        first = rows[0]
        out.append(OrderedDict([
            ("run_id", run_id), ("scope", first.get("scope", "")),
            ("n_per_type", first.get("n_per_type", "")), ("seed", first.get("seed", "")),
            ("lr", first.get("lr", "")),
            ("n_classes", len(rows)), ("n_rare(<%d)" % min_train, len(rare)),
            ("rare_classes", ";".join(f"{n}(n_train={t},f1={f:.3f})" for n, t, f in rare_rows)),
            ("macro_all", _fmt(macro_all)),
            ("macro_n>=%d" % min_train, _fmt(_mean(common))),
            ("macro_rare", _fmt(_mean(rare))),
            ("csv_macro_f1", _fmt(csv_f1)),
            ("diff_all_vs_csv", _fmt(None if csv_f1 is None else macro_all - csv_f1)),
        ]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="메인 결과 CSV 경로 (_pertype.csv 는 자동으로 찾는다)")
    ap.add_argument("--min-train", type=int, default=10, help="희소 클래스 기준 n_train (기본 10)")
    ap.add_argument("--out", default=None, help="CSV 로 저장할 경로 (없으면 표준출력)")
    args = ap.parse_args()
    rows = summarize(args.csv, args.min_train)
    if not rows:
        sys.exit("요약할 run 이 없습니다.")
    fh = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    if args.out:
        fh.close()
        print(f"{len(rows)}개 run 요약 저장: {args.out}")


if __name__ == "__main__":
    main()
