"""Rebuild the PCA arms from the same raw mwmStar spectra as the masked AE.

Fit PCA on the field subset, project every spectrum (field + members), and
write pca_64 / pca_256 parquets keyed by APOGEE_ID. Matches the masked-AE
protocol: train/embeddings on field, then project the members.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pretrain_embed import load_specs

N_COMPONENTS = 256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--field-dir", required=True)
    p.add_argument("--members-dir", required=True)
    p.add_argument("--out-64", required=True)
    p.add_argument("--out-256", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    f_ids, Xf = load_specs(Path(args.field_dir))
    m_ids, Xm = load_specs(Path(args.members_dir))
    print(f"field {len(f_ids)} members {len(m_ids)}", flush=True)

    pca = PCA(n_components=N_COMPONENTS, random_state=args.seed)
    pca.fit(Xf)
    Zf = pca.transform(Xf)
    Zm = pca.transform(Xm)
    print(f"explained var (256): {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    def write(ids, Z, k, path):
        frame = pd.DataFrame({"APOGEE_ID": ids})
        for d in range(k):
            frame[f"z{d}"] = Z[:, d].astype("float32")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        print(f"{len(frame)} x {k} -> {path}", flush=True)

    write(f_ids + m_ids, np.vstack([Zf, Zm]), 64, args.out_64)
    write(f_ids + m_ids, np.vstack([Zf, Zm]), 256, args.out_256)


if __name__ == "__main__":
    main()
