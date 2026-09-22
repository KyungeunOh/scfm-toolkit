import sys
from pathlib import Path
_REPO_ROOT = Path("/workspace")
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import scanpy as sc
from pipeline.config import load_config

cfg = load_config("/workspace/config_scfoundation.yaml")

adata = sc.read(cfg["reference_path"])
adata_test = sc.read(cfg["query_path"])

print("reference:", cfg["reference_path"])
print("  shape:", adata.shape)
print("  var_names[:5]:", list(adata.var_names[:5]))
print("  var.columns:", list(adata.var.columns))

print("query:", cfg["query_path"])
print("  shape:", adata_test.shape)
print("  var_names[:5]:", list(adata_test.var_names[:5]))
print("  var.columns:", list(adata_test.var.columns))

common = set(adata.var_names) & set(adata_test.var_names)
print("교집합 유전자 수:", len(common))
