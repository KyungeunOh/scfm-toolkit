#!/usr/bin/env python3
import csv, re, sys, collections, statistics as st

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
NAME = {"0": "freeze", "2": "last2", "4": "last4", "12": "full"}
LABEL = {"freeze": "freeze", "last2": "2-block", "last4": "4-block", "full": "full"}
ORD = ["freeze", "last2", "last4", "full"]
num = r"([0-9]*\.?[0-9]+)"


def fl(x):
    return float(x) if x not in ("", None, "None") else None


main, pt = {}, collections.defaultdict(list)
for base in sys.argv[1:]:
    for r in csv.DictReader(open(base + ".csv", encoding="utf-8")):
        main[r["run_id"]] = r
    for r in csv.DictReader(open(base + "_pertype.csv", encoding="utf-8")):
        pt[r["run_id"]].append(r)

runs, bad = [], 0
for rid, rows in pt.items():
    m = main.get(rid)
    if m is None or m["status"] != "success":
        bad += 1
        print("!! 확인 필요(메인 CSV 없음/실패):", rid)
        continue
    mm = re.search(r"ft_last(\d+)_", rows[0]["scope"])
    scope = NAME.get(mm.group(1), "last" + mm.group(1)) if mm else rows[0]["scope"]
    C = [fl(r["f1"]) for r in rows if int(r["support"]) > 0 and r["n_train"] not in ("", "0")]
    o = re.search(r"원래통계 f1=%s" % num, (m["notes"] or "").split(" | holdout")[0])
    runs.append(dict(scope=scope, n=int(rows[0]["n_per_type"]), seed=int(rows[0]["seed"]), lr=float(rows[0]["lr"]), acc=fl(m["accuracy"]),
                     A=fl(m["macro_f1"]), A0=float(o.group(1)) if o else None, C=sum(C) / len(C), gpu=m.get("gpu_name", "?")))

V = collections.defaultdict(dict)
dup = 0
for r in runs:
    k = (r["n"], r["scope"], r["lr"])
    dup += r["seed"] in V[k]
    V[k][r["seed"]] = r
print("== 점검 == run %d개, 실패/누락 %d, 중복 %d, GPU %s, (N, scope, lr) 조합 %d개" % (len(runs), bad, dup, sorted(set(r["gpu"] for r in runs)), len(V)))
print("조합별 seed 수:", sorted(collections.Counter(len(v) for v in V.values()).items()))


def ms(v):
    v = [x for x in v if x is not None]
    return "%.3f±%.3f" % (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else "-"


def paired(a, b, key="C"):
    A, B = V.get(a), V.get(b)
    if not A or not B:
        return None
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    d = [A[i][key] - B[i][key] for i in s]
    m = st.mean(d)
    h = TCRIT.get(len(d) - 1, 1.96) * st.stdev(d) / len(d) ** 0.5
    return "%+.3f [%+.3f, %+.3f] A 높음 %d/%d" % (m, m - h, m + h, sum(x > 0 for x in d), len(d))


best = {}
for n in sorted(set(k[0] for k in V)):
    print("\n" + "=" * 96)
    print("N=%d: (scope, lr) 별 평균±SD    BN 효과 = 재계산 A - 원래 A" % n)
    print("%-8s %-8s %2s | accuracy     | C            | A(runner)    | A 원래통계   | BN 효과" % ("scope", "lr", "n"))
    for k in sorted((k for k in V if k[0] == n), key=lambda k: (ORD.index(k[1]) if k[1] in ORD else 9, k[2])):
        g = list(V[k].values())
        bn = [x["A"] - x["A0"] for x in g if x["A0"] is not None]
        print("%-8s %-8g %2d | %s | %s | %s | %s | %s" % (LABEL.get(k[1], k[1]), k[2], len(g), ms([x["acc"] for x in g]), ms([x["C"] for x in g]),
              ms([x["A"] for x in g]), ms([x["A0"] for x in g]), "%+.3f" % st.mean(bn) if bn else "-"))
    print("\nscope별 C 최고 lr (사후 선택, 낙관적; 격자 끝이면 표시)")
    for sc in ORD:
        ks = [k for k in V if k[0] == n and k[1] == sc]
        if not ks:
            continue
        kb = max(ks, key=lambda k: st.mean(x["C"] for x in V[k].values()))
        lrs = sorted(k[2] for k in ks)
        edge = " (격자 끝)" if kb[2] in (lrs[0], lrs[-1]) and len(lrs) > 1 else ""
        best[(n, sc)] = kb
        print("  %-8s lr=%-7g C=%.3f%s" % (LABEL[sc], kb[2], st.mean(x["C"] for x in V[kb].values()), edge))
    print("\n같은 scope 안에서 인접 lr 차이 (큰 lr - 작은 lr, C)")
    for sc in ORD:
        lrs = sorted(k[2] for k in V if k[0] == n and k[1] == sc)
        for lo, hi in zip(lrs, lrs[1:]):
            r = paired((n, sc, hi), (n, sc, lo))
            if r:
                print("  %-8s %g - %g   %s" % (LABEL[sc], hi, lo, r))
    print("\nscope별 최고 lr끼리 (사후 선택, 낙관적) (C)")
    for a, b in (("last2", "freeze"), ("last4", "freeze"), ("full", "freeze"), ("last4", "last2"), ("full", "last2"), ("full", "last4")):
        if (n, a) in best and (n, b) in best:
            r = paired(best[(n, a)], best[(n, b)])
            if r:
                print("  %-8s - %-8s %s" % (LABEL[a], LABEL[b], r))
