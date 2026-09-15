"""
tests/test_benchmark.py
benchmark/grid.py, benchmark/run_log.py 중 torch/scGPT 없이도 검증 가능한 부분만
확인한다(tests/test_pipeline.py와 같은 스타일 - pytest 대신 순수 스크립트).
GPU/모델 관련 부분(memory_probe.py, run_*_sweep.py, scfoundation_adapter.py)은
여기서 검증할 수 없다 - benchmark/README.md "검증 상태" 절 참고.
python3 tests/test_benchmark.py 로 실행.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.grid import coarse_grid, default_smoke_grid, effective_batch_size, refine_between
from benchmark.run_log import RUN_LOG_COLUMNS, append_row, build_row, log_run


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def expect(cond: bool, label: str):
    print(f"{'✅ PASS' if cond else '❌ FAIL'} - {label}")


# ---------------------------------------------------------------------
section("1. grid.py - default_smoke_grid")
smoke = default_smoke_grid()
expect(len(smoke) == 5, f"smoke grid 5개 (실제 {len(smoke)}개)")
expect(all(set(d.keys()) == {"precision", "micro_batch_size", "grad_accum_steps",
                              "activation_checkpointing", "max_seq_len"} for d in smoke),
       "smoke grid의 모든 항목이 5개 축을 다 갖고 있음")

section("2. grid.py - coarse_grid 조합 수")
coarse = coarse_grid(precisions=["fp16", "bf16"], micro_batch_sizes=[1, 8],
                      grad_accum_steps=[1], activation_checkpointing=[False, True],
                      max_seq_lens=[1500])
expect(len(coarse) == 2 * 2 * 1 * 2 * 1, f"2x2x1x2x1=8개 (실제 {len(coarse)}개)")
expect(len(coarse) == len({tuple(sorted(d.items())) for d in coarse}), "중복 조합 없음")

section("3. grid.py - refine_between 정수 축")
refined = refine_between("micro_batch_size", 16, 64, n_points=3)
values = [d["micro_batch_size"] for d in refined]
expect(all(16 < v < 64 for v in values), f"모든 값이 16과 64 사이 (실제 {values})")
expect(len(values) == len(set(values)), "중복 값 없음")

section("4. grid.py - refine_between 비정수 축 거부")
try:
    refine_between("precision", "fp16", "bf16")
    print("❌ FAIL - TypeError가 발생해야 함")
except TypeError:
    print("✅ PASS - 범주형 축은 TypeError로 명확히 거부됨")

section("5. grid.py - effective_batch_size")
expect(effective_batch_size({"micro_batch_size": 8, "grad_accum_steps": 4}) == 32,
       "8 x 4 = 32")
expect(effective_batch_size({}) is None, "필드 없으면 None")

# ---------------------------------------------------------------------
section("6. logging.py - build_row는 정의된 컬럼만 허용")
row = build_row(model="scgpt", precision="fp16")
expect(row["model"] == "scgpt" and row["precision"] == "fp16", "지정한 값 반영")
expect(set(row.keys()) == set(RUN_LOG_COLUMNS), "모든 RUN_LOG_COLUMNS가 채워짐(미지정은 빈 문자열)")
try:
    build_row(no_such_column="x")
    print("❌ FAIL - ValueError가 발생해야 함")
except ValueError:
    print("✅ PASS - 스키마에 없는 필드는 ValueError로 거부됨")

section("7. logging.py - append_row: 헤더 생성 + 이어쓰기")
with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / "runs.csv"
    log_run(csv_path, model="scgpt", precision="fp16", status="success")
    log_run(csv_path, model="scgpt", precision="bf16", status="oom")
    lines = csv_path.read_text().splitlines()
    expect(len(lines) == 3, f"헤더 1줄 + 데이터 2줄 = 3줄 (실제 {len(lines)}줄)")
    expect(lines[0] == ",".join(RUN_LOG_COLUMNS), "헤더가 RUN_LOG_COLUMNS 순서와 일치")

section("8. logging.py - 헤더 불일치 시 명확히 거부")
with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / "runs.csv"
    csv_path.write_text("run_id,some_old_column\nabc,1\n")
    try:
        log_run(csv_path, model="scgpt")
        print("❌ FAIL - RuntimeError가 발생해야 함")
    except RuntimeError:
        print("✅ PASS - 스키마가 바뀐 기존 CSV에 이어쓰려 하면 RuntimeError로 명확히 거부됨")

print("\n" + "=" * 70)
print("완료 - GPU/torch/scGPT/scFoundation 관련 부분은 이 테스트로 검증되지 않음")
print("(benchmark/README.md '검증 상태' 절 참고, 서버에서 직접 확인 필요)")
print("=" * 70)
