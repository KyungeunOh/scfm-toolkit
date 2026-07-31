"""
pipeline/config.py, pipeline/data_io.py를 scGPT/torch 없이 검증.
python3 tests/test_pipeline.py 로 실행.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.config import ConfigError, validate_config
from pipeline.data_io import DataValidationError, validate_h5ad

FIXTURES = Path(__file__).parent / "fixtures"


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def expect_fail(fn, exc_type, label):
    try:
        fn()
        print(f"❌ FAIL (예상: {exc_type.__name__} 발생해야 함, 실제: 통과됨) - {label}")
    except exc_type as e:
        print(f"✅ PASS - {label}\n   메시지: {e}")
    except Exception as e:
        print(f"❌ FAIL (엉뚱한 예외: {type(e).__name__}: {e}) - {label}")


def expect_pass(fn, label):
    try:
        fn()
        print(f"✅ PASS - {label}")
    except Exception as e:
        print(f"❌ FAIL ({type(e).__name__}: {e}) - {label}")


# ---------------------------------------------------------------------
section("1. config validation - 정상 케이스")
good_cfg = {
    "mode": "finetune_predict",
    "output_dir": "/tmp/out",
    "reference_path": str(FIXTURES / "reference.h5ad"),
    "query_path": str(FIXTURES / "query.h5ad"),
    "model_dir": str(FIXTURES / "model"),
}
expect_pass(
    lambda: validate_config(good_cfg, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir"]),
    "필수 키/경로 모두 있는 config",
)

section("2. config validation - 필수 키 누락 (Before: KeyError traceback / After: 이런 메시지)")
bad_cfg_missing_key = {"mode": "finetune_predict", "output_dir": "/tmp/out"}
expect_fail(
    lambda: validate_config(bad_cfg_missing_key, adapter_required_keys=["reference_path", "query_path", "model_dir"]),
    ConfigError, "필수 키 reference_path 누락",
)

section("3. config validation - 존재하지 않는 경로")
bad_cfg_path = dict(good_cfg, reference_path="/no/such/file.h5ad")
expect_fail(
    lambda: validate_config(bad_cfg_path, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir"]),
    ConfigError, "reference_path가 실제로 존재하지 않는 파일",
)

section("4. config validation - 아직 구현 안 된 mode")
bad_cfg_mode = dict(good_cfg, mode="embed")
expect_fail(
    lambda: validate_config(bad_cfg_mode, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir"]),
    ConfigError, "mode=embed (config 구조상 자리는 있지만 미구현)",
)

# ---------------------------------------------------------------------
section("5. h5ad validation - 정상 케이스 (label column 있음, vocab 매칭 OK)")
vocab_genes = {f"GENE{i}" for i in range(150)}  # make_synthetic_data.py와 동일 로직
expect_pass(
    lambda: validate_h5ad(str(FIXTURES / "reference.h5ad"), "Reference", celltype_col="celltype",
                           vocab_genes=vocab_genes),
    "reference.h5ad, celltype_col 있음, vocab 매칭 150/200=75%",
)

section("6. h5ad validation - label column이 obs에 없음")
expect_fail(
    lambda: validate_h5ad(str(FIXTURES / "reference.h5ad"), "Reference", celltype_col="no_such_column",
                           vocab_genes=vocab_genes),
    DataValidationError, "obs에 없는 celltype_col 지정",
)

section("7. h5ad validation - vocab 매칭률이 기준 미달일 때")
tiny_vocab = {"GENE0", "GENE1"}  # 200개 중 2개만 매칭 = 1%
expect_fail(
    lambda: validate_h5ad(str(FIXTURES / "reference.h5ad"), "Reference", celltype_col="celltype",
                           vocab_genes=tiny_vocab),
    DataValidationError, "vocab 매칭률 1% (최소 기준 30% 미달)",
)

section("8. h5ad validation - 파일이 아예 없음")
expect_fail(
    lambda: validate_h5ad("/no/such/file.h5ad", "Reference", celltype_col="celltype"),
    DataValidationError, "존재하지 않는 h5ad 경로",
)

print("\n모든 테스트 완료.")
