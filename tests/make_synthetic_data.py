"""테스트용 합성 h5ad + mock vocab.json / args.json 생성.
실제 scGPT 가중치/데이터 없이 pipeline 레이어(validation, report)를 검증하기 위한 것.
"""
import json
import numpy as np
import anndata as ad
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

genes = [f"GENE{i}" for i in range(200)]
celltypes = ["T_cell", "B_cell", "NK_cell", "Monocyte"]


def make_adata(n_cells, label_col="celltype"):
    X = rng.poisson(2, size=(n_cells, len(genes))).astype(np.float32)
    obs = pd.DataFrame({
        label_col: rng.choice(celltypes, size=n_cells),
    })
    var = pd.DataFrame({"gene_name": genes}, index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)


ref = make_adata(300)
query = make_adata(100)

ref.write_h5ad(OUT / "reference.h5ad")
query.write_h5ad(OUT / "query.h5ad")

# mock model dir: vocab.json은 GeneVocab.from_file이 읽는 형식(token -> id) 그대로 흉내
model_dir = OUT / "model"
model_dir.mkdir(exist_ok=True)
vocab = {g: i for i, g in enumerate(genes[:150])}  # 절반만 겹치게 해서 매칭률 테스트
vocab.update({"<pad>": 998, "<cls>": 999, "<eoc>": 1000})
with open(model_dir / "vocab.json", "w") as f:
    json.dump(vocab, f)
with open(model_dir / "args.json", "w") as f:
    json.dump({"embsize": 64, "nheads": 4, "d_hid": 64, "nlayers": 2}, f)
(model_dir / "best_model.pt").write_bytes(b"fake")  # 존재 여부만 체크하므로 내용 불필요

print("합성 데이터 생성 완료:", OUT)
