"""
Embedding-based candidate generation, to UNION with existing token blocking
(not replace it).

Run per split, one source table per process invocation -- this bounds peak
memory (each process only ever holds one country's data) and lets you
resume without redoing countries that already finished:

    python3 embed_block.py train source2
    python3 embed_block.py train source3
    python3 embed_block.py test source2
    python3 embed_block.py test source3

(Running with just `python3 embed_block.py train` still works and loops
over both tables in one process, but prefer the one-table-per-process form
above for large runs -- it's what avoids the OOM kill.)

Produces one parquet file PER COUNTRY:
    artifacts/blocking/{split}_{tbl}_{country}_embed_pairs.parquet
(unless a combined artifacts/blocking/{split}_{tbl}_embed_pairs.parquet
from an earlier run already exists, in which case that table is skipped
entirely.)

Read all of a table's country files together later with a glob, e.g.:
    read_parquet('artifacts/blocking/train_source3_*_embed_pairs.parquet')
"""
import gc
import sys
import time
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import hnswlib

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "normalized"
OUT = ROOT / "artifacts" / "blocking"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 100
BATCH_SIZE = 512

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}", flush=True)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)


def encode(names):
    return model.encode(names, batch_size=BATCH_SIZE, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)


def get_countries(con, split):
    countries = set()
    for src in ["source1", "source2", "source3"]:
        rows = con.execute(
            f"select distinct country from read_parquet('{ART}/{split}_{src}_normalized.parquet')"
        ).fetchall()
        countries.update(r[0] for r in rows)
    return sorted(countries)


def safe(country):
    return "".join(c if c.isalnum() else "_" for c in country)


def embed_block(split, only_tbl=None):
    con = duckdb.connect()
    countries = get_countries(con, split)
    print(f"countries: {countries}", flush=True)

    tables = [only_tbl] if only_tbl else ["source2", "source3"]

    for tbl in tables:
        legacy_path = OUT / f"{split}_{tbl}_embed_pairs.parquet"
        if legacy_path.exists():
            print(f"  {tbl}: {legacy_path.name} already exists from an earlier "
                  f"run -- skipping this table entirely", flush=True)
            continue

        for country in countries:
            out_path = OUT / f"{split}_{tbl}_{safe(country)}_embed_pairs.parquet"
            if out_path.exists():
                print(f"  {country} vs {tbl}: {out_path.name} already exists -- skipping", flush=True)
                continue

            esc = country.replace("'", "''")
            t0 = time.time()

            s1 = con.execute(f"""
                select entity_id, business_name
                from read_parquet('{ART}/{split}_source1_normalized.parquet')
                where country = '{esc}'
            """).df()
            o = con.execute(f"""
                select entity_id, business_name
                from read_parquet('{ART}/{split}_{tbl}_normalized.parquet')
                where country = '{esc}'
            """).df()

            if len(s1) == 0 or len(o) == 0:
                print(f"  {country}: skip (s1={len(s1)}, o={len(o)})", flush=True)
                continue

            print(f"  {country} vs {tbl}: s1={len(s1)} o={len(o)}", flush=True)
            s1_emb = encode(s1["business_name"].fillna("").tolist())
            o_emb = encode(o["business_name"].fillna("").tolist())

            dim = o_emb.shape[1]
            k = min(TOP_K, len(o))
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(o), ef_construction=200, M=16)
            index.add_items(o_emb, np.arange(len(o)))
            index.set_ef(max(50, k * 2))

            labels, distances = index.knn_query(s1_emb, k=k)
            del s1_emb, o_emb, index

            o_ids = o["entity_id"].to_numpy()
            s1_ids = s1["entity_id"].to_numpy()

            # Fully vectorized (numpy) -- no per-pair Python tuple/float
            # objects. The old version built a single Python list of
            # ~220M (str, str, float) tuples across BOTH source tables in
            # one process before ever writing anything out; that list plus
            # the DataFrame built from it needed tens of GB of RAM at
            # once, which is what killed the process right at the finish
            # line last run. This version keeps at most one country's
            # worth of flat numpy arrays in memory and writes them
            # immediately.
            s1_col = np.repeat(s1_ids, k)
            o_col = o_ids[labels.reshape(-1)]
            score_col = (1 - distances).reshape(-1).astype(np.float32)
            del labels, distances, o_ids, s1_ids

            df = pd.DataFrame({"s1_id": s1_col, "o_id": o_col, "embed_score": score_col})
            del s1_col, o_col, score_col
            df.to_parquet(out_path, index=False)
            n_rows = len(df)
            del df
            gc.collect()

            print(f"    done in {time.time()-t0:.1f}s, saved {out_path.name}: {n_rows} rows", flush=True)


if __name__ == "__main__":
    split = sys.argv[1]
    only_tbl = sys.argv[2] if len(sys.argv) > 2 else None
    embed_block(split, only_tbl)
