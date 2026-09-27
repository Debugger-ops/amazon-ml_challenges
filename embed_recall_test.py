"""
Quick validation: would embedding-based nearest-neighbor retrieval recover the
true matches that our current exact-token blocking completely misses?

Run this from code/business_entity_resolution/ in your own Mac terminal
(with the venv activated), NOT via the sandboxed device shell -- it needs
real CPU/GPU to be fast.

    pip install sentence-transformers
    python3 embed_recall_test.py
"""
import duckdb
import numpy as np
import pandas as pd
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # student_resource/
ART = ROOT / "artifacts" / "normalized"
BLOCK = ROOT / "artifacts" / "blocking"
TRAIN_GT = ROOT / "dataset" / "train" / "train_ground_truth.tsv"

random.seed(42)
np.random.seed(42)

N_SAMPLE_ENTITIES = 200      # S1 entities to test recall for
POOL_SIZE = 20000            # random decoy O records per source
TOP_KS = [20, 50, 100]

con = duckdb.connect()

print("Loading ground truth and finding pairs missed by current blocking...", flush=True)

# Explode ground truth into (s1_id, o_id) pairs
gt = con.execute(f"""
    with exploded as (
        select source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as o_id
        from read_csv('{TRAIN_GT}', delim='\t', header=true)
        where matched_entity_ids is not null and matched_entity_ids != ''
    )
    select s1_id, o_id from exploded
""").df()

results = {}
for src, prefix in [("source2", "S2-"), ("source3", "S3-")]:
    gt_src = gt[gt["o_id"].str.startswith(prefix)]
    missed = con.execute(f"""
        select g.s1_id, g.o_id
        from gt_src g
        anti join read_parquet('{BLOCK}/train_{src}_pairs_capped.parquet') b
        on g.s1_id = b.s1_id and g.o_id = b.o_id
    """).df()
    print(f"{src}: {len(missed)} missed true pairs (out of {len(gt_src)} total)", flush=True)

    # Sample N distinct S1 entities from the missed set
    sample_s1 = missed["s1_id"].drop_duplicates().sample(
        n=min(N_SAMPLE_ENTITIES, missed["s1_id"].nunique()), random_state=42
    ).tolist()
    sample_missed = missed[missed["s1_id"].isin(sample_s1)]

    # Get country + name for the sampled S1 entities
    s1_info = con.execute(f"""
        select entity_id as s1_id, business_name, country
        from read_parquet('{ART}/train_source1_normalized.parquet')
        where entity_id in ({",".join("'" + x + "'" for x in sample_s1)})
    """).df()

    # Build candidate pool: random decoys + guaranteed true targets
    true_o_ids = sample_missed["o_id"].unique().tolist()
    decoys = con.execute(f"""
        select entity_id as o_id, business_name
        from read_parquet('{ART}/train_{src}_normalized.parquet')
        using sample {POOL_SIZE} rows
    """).df()
    true_o_info = con.execute(f"""
        select entity_id as o_id, business_name
        from read_parquet('{ART}/train_{src}_normalized.parquet')
        where entity_id in ({",".join("'" + x + "'" for x in true_o_ids)})
    """).df()
    pool = pd.concat([decoys, true_o_info]).drop_duplicates(subset="o_id").reset_index(drop=True)

    results[src] = dict(s1_info=s1_info, sample_missed=sample_missed, pool=pool)
    print(f"  sampled {len(sample_s1)} S1 entities, candidate pool size {len(pool)}", flush=True)

print("\nLoading embedding model (multilingual, Apache 2.0, ~118M params)...", flush=True)
from sentence_transformers import SentenceTransformer
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}", flush=True)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

for src, data in results.items():
    print(f"\n=== {src} ===", flush=True)
    s1_info, sample_missed, pool = data["s1_info"], data["sample_missed"], data["pool"]

    s1_emb = model.encode(s1_info["business_name"].tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    pool_emb = model.encode(pool["business_name"].tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    pool_ids = pool["o_id"].tolist()
    pool_idx = {oid: i for i, oid in enumerate(pool_ids)}

    sim = s1_emb @ pool_emb.T  # cosine similarity (both normalized)

    hits = {k: 0 for k in TOP_KS}
    total = 0
    for i, s1_id in enumerate(s1_info["s1_id"]):
        true_targets = set(sample_missed[sample_missed["s1_id"] == s1_id]["o_id"])
        true_targets = {t for t in true_targets if t in pool_idx}
        if not true_targets:
            continue
        total += 1
        order = np.argsort(-sim[i])
        for k in TOP_KS:
            top_k_ids = {pool_ids[j] for j in order[:k]}
            if top_k_ids & true_targets:
                hits[k] += 1

    print(f"Evaluated {total} S1 entities (with true target in pool)", flush=True)
    for k in TOP_KS:
        pct = 100 * hits[k] / total if total else 0
        print(f"  recall@{k}: {hits[k]}/{total} = {pct:.1f}%", flush=True)
