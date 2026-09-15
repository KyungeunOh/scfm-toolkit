"""
benchmark/summarize_results.py
run_scgpt_sweep.py / run_scfoundation_sweep.py가 쌓은 CSV를 사람이 읽기 좋은
표로 요약한다. pandas만 있으면 되고(이 프로젝트 requirements.txt에 이미 포함),
GPU/torch는 필요 없다 - 결과 확인은 서버가 아니어도(예: 이 CSV만 로컬로 복사해와서)
할 수 있다.

사용법:
  python3 benchmark/summarize_results.py benchmark_results/scgpt_sweep_smoke.csv
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if df.empty:
        print("CSV에 행이 없습니다.")
        sys.exit(0)

    cols = [
        "precision", "micro_batch_size", "grad_accum_steps", "activation_checkpointing",
        "max_seq_len", "status", "peak_allocated_mb", "peak_reserved_mb",
        "total_seconds", "accuracy", "macro_f1",
    ]
    cols = [c for c in cols if c in df.columns]
    print("=" * 100)
    print(f"{args.csv_path}  ({len(df)}개 실행 결과)")
    print("=" * 100)
    print(df[cols].to_string(index=False))

    print("\n" + "-" * 100)
    print("상태별 개수 (success=성공 / oom=예상된 실험 결과 / error=코드 버그 의심 - error_message 컬럼 확인)")
    print("-" * 100)
    print(df["status"].value_counts().to_string())

    error_rows = df[df["status"] == "error"]
    if not error_rows.empty:
        print("\n" + "!" * 100)
        print(f"status=error {len(error_rows)}건 - 실험 실패가 아니라 코드/설정 문제일 가능성이 높습니다:")
        print("!" * 100)
        for _, row in error_rows.iterrows():
            print(f"- run_id={row['run_id']}: {str(row['error_message'])[:300]}")

    ok = df[df["status"] == "success"]
    if {"activation_checkpointing", "peak_allocated_mb"}.issubset(ok.columns) and not ok.empty:
        grouped = ok.groupby("activation_checkpointing")["peak_allocated_mb"].agg(["mean", "min", "max", "count"])
        if len(grouped) > 1:
            print("\n" + "-" * 100)
            print("activation_checkpointing True/False별 peak_allocated_mb 비교")
            print("(True 쪽 평균이 False보다 뚜렷이 낮아야 checkpointing이 실제로 동작한 것)")
            print("-" * 100)
            print(grouped.to_string())


if __name__ == "__main__":
    main()
