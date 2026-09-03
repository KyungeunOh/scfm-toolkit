"""
pipeline/config.py, pipeline/data_io.py, pipeline/report.py의 label-optional
부분을 scGPT/torch 없이 검증. python3 tests/test_pipeline.py 로 실행.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

import numpy as np

from pipeline.config import ConfigError, resolve_run_output_dir, validate_config
from pipeline.data_io import DataValidationError, load_h5ad_full, validate_h5ad
from pipeline.reference_mapping import knn_label_transfer
from pipeline.report import flag_low_confidence, save_metrics

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
bad_cfg_mode = dict(good_cfg, mode="train_head")
expect_fail(
    lambda: validate_config(bad_cfg_mode, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir"]),
    ConfigError, "mode=train_head (config 구조상 자리는 있지만 아직 미구현)",
)

section("4-1. config validation - mode: embed (reference mapping, 구현됨)")
embed_cfg = dict(good_cfg, mode="embed")
expect_pass(
    lambda: validate_config(embed_cfg, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir"]),
    "mode=embed은 이제 정상적으로 실행 가능해야 함",
)

section("4-2. config validation - mode: integration (zero-shot batch 통합, 구현됨)")
# ScGPTAdapter.extra_required_config_keys("integration")이 반환하는 것과 동일한 키
# (data_path, batch_key)를 finetune_predict용 키에 추가해서 검증 - run.py의
# `adapter.required_config_keys + adapter.extra_required_config_keys(mode)` 조합
# 로직 자체를 흉내낸다 (실제 ScGPTAdapter는 torch를 import해서 이 개발 환경에서는
# 인스턴스화할 수 없음 - adapters/base.py, adapters/scgpt_adapter.py의 메서드
# 시그니처는 별도로 py_compile + ABC 스텁으로 검증).
integration_cfg = dict(
    good_cfg, mode="integration",
    data_path=str(FIXTURES / "reference.h5ad"),
    batch_key="sample",
)
integration_required = ["reference_path", "query_path", "model_dir", "data_path", "batch_key"]
integration_paths = ["reference_path", "query_path", "model_dir", "data_path"]
expect_pass(
    lambda: validate_config(integration_cfg, adapter_required_keys=integration_required,
                             adapter_path_keys=integration_paths),
    "mode=integration은 정상적으로 실행 가능해야 함",
)

section("4-3. config validation - mode: integration인데 batch_key가 없음")
integration_cfg_missing_batch_key = dict(good_cfg, mode="integration", data_path=str(FIXTURES / "reference.h5ad"))
expect_fail(
    lambda: validate_config(integration_cfg_missing_batch_key,
                             adapter_required_keys=integration_required,
                             adapter_path_keys=integration_paths),
    ConfigError, "mode=integration인데 batch_key가 config.yaml에 없음",
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

# ---------------------------------------------------------------------
# finetuned_model_path (모델 재사용) 관련 config validation
section("9. config validation - finetuned_model_path 미지정(null)은 선택 항목이라 통과")
cfg_no_reuse = dict(good_cfg, finetuned_model_path=None)
expect_pass(
    lambda: validate_config(cfg_no_reuse, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir", "finetuned_model_path"]),
    "finetuned_model_path: null (선택 항목, 값 없으면 검증을 건너뜀)",
)

section("10. config validation - finetuned_model_path를 지정했는데 파일이 없으면 실패")
cfg_bad_reuse = dict(good_cfg, finetuned_model_path="/no/such/finetuned_model.pt")
expect_fail(
    lambda: validate_config(cfg_bad_reuse, adapter_required_keys=["reference_path", "query_path", "model_dir"],
                             adapter_path_keys=["reference_path", "query_path", "model_dir", "finetuned_model_path"]),
    ConfigError, "finetuned_model_path가 지정됐지만 파일이 존재하지 않음",
)

# ---------------------------------------------------------------------
# 예측 신뢰도 경고 (low_confidence) 관련
section("11. 예측 신뢰도 플래그 - low_confidence 컬럼/통계 계산")


class _FakeAdata:
    """flag_low_confidence/save_metrics는 adata.obs(pandas DataFrame)만 사용하므로,
    실제 AnnData 없이 이 정도 stub으로 충분히 검증 가능."""

    def __init__(self, obs):
        self.obs = obs


def _check_confidence_summary():
    fake = _FakeAdata(pd.DataFrame({"pred_score": [0.9, 0.4, 0.55, 0.2]}))
    summary = flag_low_confidence(fake, 0.5)
    expected = {
        "low_confidence_threshold": 0.5,
        "low_confidence_count": 2,
        "low_confidence_total": 4,
        "low_confidence_ratio": 0.5,
    }
    assert summary == expected, f"summary={summary}"
    assert list(fake.obs["low_confidence"]) == [False, True, False, True], list(fake.obs["low_confidence"])


expect_pass(_check_confidence_summary, "threshold=0.5일 때 4개 중 2개(0.4, 0.2)가 low_confidence로 표시됨")


def _check_confidence_disabled():
    fake = _FakeAdata(pd.DataFrame({"pred_score": [0.9, 0.1]}))
    summary = flag_low_confidence(fake, None)
    assert summary is None, f"summary={summary}"
    assert "low_confidence" not in fake.obs.columns


expect_pass(_check_confidence_disabled, "threshold=None이면 비활성화 (None 반환, obs 변경 없음)")

section("12. metrics.json - label 없이도 confidence 통계만으로 저장됨 (예측 전용 실행)")


def _check_metrics_confidence_only():
    fake = _FakeAdata(pd.DataFrame({
        "predictions": ["T_cell", "B_cell"],
        "pred_score": [0.9, 0.3],
    }))
    confidence_summary = flag_low_confidence(fake, 0.5)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        metrics = save_metrics(fake, out_dir, celltype_col=None, confidence_summary=confidence_summary)
        assert metrics is not None
        assert "accuracy" not in metrics, "label이 없으면 accuracy는 계산되지 않아야 함"
        assert metrics["low_confidence_count"] == 1
        assert (out_dir / "metrics.json").exists()
        assert not (out_dir / "per_class_metrics.csv").exists(), "label 없으면 per-class 파일도 안 만들어야 함"


expect_pass(_check_metrics_confidence_only, "label 없어도 low_confidence 통계는 metrics.json에 저장됨")

# ---------------------------------------------------------------------
# mode: embed (reference mapping) 관련 - load_h5ad_full, knn_label_transfer
section("13. data_io.load_h5ad_full - 전체 데이터를 실제로 메모리에 로드")


def _check_load_h5ad_full():
    adata = load_h5ad_full(str(FIXTURES / "reference.h5ad"))
    assert adata.n_obs > 0 and adata.n_vars > 0
    assert adata.X is not None, "backed 모드가 아니라 X가 실제로 로드돼 있어야 함"


expect_pass(_check_load_h5ad_full, "reference.h5ad를 전체 로드 (validate_h5ad의 backed 모드와 달리 X 접근 가능)")

section("14. reference_mapping.knn_label_transfer - 임베딩 공간 k-NN 다수결 매핑")


def _check_knn_label_transfer():
    rng = np.random.RandomState(0)
    centers = {"T cell": [0, 0], "B cell": [10, 10], "NK cell": [0, 10]}
    ref_emb, ref_labels = [], []
    for label, c in centers.items():
        pts = rng.normal(loc=c, scale=0.5, size=(30, 2))
        ref_emb.append(pts)
        ref_labels.extend([label] * 30)
    ref_emb = np.vstack(ref_emb)
    ref_labels = np.array(ref_labels)

    query_emb = np.array([[0.1, 0.1], [10.1, 9.9], [0.2, 9.8]])
    preds, scores = knn_label_transfer(ref_emb, ref_labels, query_emb, k=10)

    assert list(preds) == ["T cell", "B cell", "NK cell"], list(preds)
    assert all(s == 1.0 for s in scores), f"명확히 분리된 클러스터는 만장일치(score=1.0)여야 함: {scores}"


expect_pass(_check_knn_label_transfer, "3개의 뚜렷한 클러스터에 대해 k-NN 다수결로 정확히 매핑됨")


def _check_knn_k_larger_than_reference():
    ref_emb = np.array([[0.0, 0.0], [1.0, 1.0]])
    ref_labels = np.array(["A", "B"])
    query_emb = np.array([[0.0, 0.0]])
    preds, scores = knn_label_transfer(ref_emb, ref_labels, query_emb, k=10000)
    assert len(preds) == 1, "k가 reference 수보다 커도 에러 없이 clip돼야 함"


expect_pass(_check_knn_k_larger_than_reference, "k > len(reference)일 때도 안전하게 처리됨")

# ---------------------------------------------------------------------
section("15. config.resolve_run_output_dir - model/mode/날짜별로 output 폴더 정리")


def _check_resolve_output_dir_basic():
    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        result = resolve_run_output_dir(tmp, "scgpt", "embed", today=date(2026, 9, 4))
        assert result == Path(tmp) / "scgpt" / "embed_260904", result
        assert not result.exists(), "실행 전이므로 아직 폴더가 실제로 생기면 안 됨(경로만 정함)"


expect_pass(_check_resolve_output_dir_basic, "base/model/mode_YYMMDD 형태로 경로를 정함 (아직 폴더는 안 만듦)")


def _check_resolve_output_dir_collision():
    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        first = resolve_run_output_dir(tmp, "scgpt", "embed", today=date(2026, 9, 4))
        first.mkdir(parents=True)  # 같은 날 같은 model+mode로 "이미 실행한 적 있음"을 흉내
        second = resolve_run_output_dir(tmp, "scgpt", "embed", today=date(2026, 9, 4))
        assert second == Path(tmp) / "scgpt" / "embed_260904_2", second
        second.mkdir(parents=True)
        third = resolve_run_output_dir(tmp, "scgpt", "embed", today=date(2026, 9, 4))
        assert third == Path(tmp) / "scgpt" / "embed_260904_3", third


expect_pass(_check_resolve_output_dir_collision, "같은 날 같은 model+mode를 다시 실행하면 기존 결과를 안 덮어쓰고 _2, _3 ... 을 붙임")


def _check_resolve_output_dir_different_mode_no_collision():
    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        embed_dir = resolve_run_output_dir(tmp, "scgpt", "embed", today=date(2026, 9, 4))
        embed_dir.mkdir(parents=True)
        integration_dir = resolve_run_output_dir(tmp, "scgpt", "integration", today=date(2026, 9, 4))
        assert integration_dir == Path(tmp) / "scgpt" / "integration_260904", integration_dir
        assert not integration_dir.exists(), "mode가 다르면 서로 충돌하면 안 됨"


expect_pass(_check_resolve_output_dir_different_mode_no_collision, "mode가 다르면 같은 날이어도 별도 폴더 (충돌 없음)")

print("\n모든 테스트 완료.")
