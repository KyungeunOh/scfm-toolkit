import sys
from pathlib import Path
_REPO_ROOT = Path("/workspace")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import torch
from adapters.scfoundation_adapter import ScFoundationAdapter
from pipeline.config import load_config
from benchmark.run_scfoundation_sweep import apply_overrides, _build_ctx

cfg_base = load_config("/workspace/config_scfoundation.yaml")
adapter = ScFoundationAdapter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx = _build_ctx(cfg_base, adapter, quiet=False)

override = {"precision": "fp16", "micro_batch_size": 1, "grad_accum_steps": 1,
            "activation_checkpointing": False, "max_seq_len": 500}
cfg = apply_overrides(ctx["cfg_base"], override)

print(">>> adata_raw shape (세포 x 유전자):", ctx["adata_raw"].shape)
print(">>> cfg n_hvg_genes:", cfg.get("n_hvg_genes"))
print(">>> preprocess() 호출")
adata = adapter.preprocess(ctx["adata_raw"].copy(), cfg)
print(">>> preprocess() 성공, adata shape:", adata.shape)

print(">>> prepare_inputs() 호출")
prepared = adapter.prepare_inputs(adata, cfg)
print(">>> prepare_inputs() 성공")

print(">>> load_model() 호출")
model = adapter.load_model(cfg, ctx["num_types"], device)
print(">>> load_model() 성공")
