#!/usr/bin/env python3
"""benchmark_results/geneformer_smoke/smoke_*/smoke_result.json 을 표로 요약한다.
predictions 해시가 같으면 셀별 예측(pred_score 포함)이 완전히 같다는 뜻이다.
사용: python3 benchmark/tools/summarize_smoke.py [결과 폴더]"""
import glob, hashlib, json, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/scfm-toolkit/benchmark_results/geneformer_smoke")
cols = ["started_utc", "ok", "lr", "epochs", "seed", "n_per_type", "n_query", "macro_f1_default", "accuracy",
        "peak_alloc_mb", "peak_resv_mb", "pred_sha", "git_commit"]
print("\t".join(cols))
for f in sorted(glob.glob(os.path.join(root, "smoke_*", "smoke_result.json"))):
    r = json.load(open(f, encoding="utf-8")); sc = r.get("score") or {}
    pf = os.path.join(os.path.dirname(f), "predictions.csv")
    sha = hashlib.sha256(open(pf, "rb").read()).hexdigest()[:10] if os.path.exists(pf) else ""
    row = [r.get("started_utc", ""), r.get("ok", ""), r.get("lr", ""), r.get("epochs", ""), r.get("seed", ""),
           r.get("n_per_type", ""), sc.get("n_query", r.get("n_query", "")), sc.get("macro_f1_default", ""),
           sc.get("accuracy", ""), r.get("peak_allocated_mb", ""), r.get("peak_reserved_mb", ""), sha, r.get("git_commit", "")]
    print("\t".join(str(x) for x in row))
