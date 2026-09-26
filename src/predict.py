import duckdb, pandas as pd, numpy as np, lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FEAT = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "model"
ART = ROOT / "artifacts" / "normalized"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["source","by_name_token","by_addr_token","name_jw_full","name_jw_core",
            "name_lev_sim","name_tok_jaccard","addr_jw_full","addr_lev_sim",
            "addr_tok_jaccard","pin_match","legal_marker_match","name_len_diff"]

CAND_FLOOR = 0.1
BATCH_SIZE = 1_000_000


def main():
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
    threshold = float((MODEL_DIR / "threshold.txt").read_text().strip())

    tmp_cand = FEAT / "_tmp_candidates.parquet"
    tmp_match = FEAT / "_tmp_matches.parquet"
    for p in (tmp_cand, tmp_match):
        if p.exists():
            p.unlink()

    # Stream through the feature file so we never hold all ~117M rows in memory
    # at once: predict per-batch, immediately discard rows below the candidate
    # floor, and append survivors to small on-disk parquet files.
    pf = pq.ParquetFile(FEAT / "test_features_final.parquet")
    cand_writer = None
    match_writer = None
    total = 0
    n_cand = 0
    n_match = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["s1_id", "o_id"] + FEATURES):
        chunk = batch.to_pandas()
        X = chunk[FEATURES].astype("float32")
        proba = model.predict(X)
        ids = chunk[["s1_id", "o_id"]]

        cand_mask = proba >= CAND_FLOOR
        if cand_mask.any():
            tbl = pa.Table.from_pandas(ids.loc[cand_mask], preserve_index=False)
            if cand_writer is None:
                cand_writer = pq.ParquetWriter(str(tmp_cand), tbl.schema)
            cand_writer.write_table(tbl)
            n_cand += int(cand_mask.sum())

        match_mask = proba >= threshold
        if match_mask.any():
            tbl2 = pa.Table.from_pandas(ids.loc[match_mask], preserve_index=False)
            if match_writer is None:
                match_writer = pq.ParquetWriter(str(tmp_match), tbl2.schema)
            match_writer.write_table(tbl2)
            n_match += int(match_mask.sum())

        total += len(chunk)
        print(f"processed {total} rows | candidates so far {n_cand} | matches so far {n_match}", flush=True)

    if cand_writer is not None:
        cand_writer.close()
    if match_writer is not None:
        match_writer.close()

    def build(tmp_path, out_name, id_col):
        if tmp_path.exists():
            grouped_sql = f"""
                select s1_id, string_agg(o_id, ',') as {id_col}
                from (select distinct s1_id, o_id from read_parquet('{tmp_path}'))
                group by s1_id
            """
        else:
            grouped_sql = f"select cast(null as varchar) as s1_id, cast(null as varchar) as {id_col} where 1=0"
        query = f"""
            copy (
                select a.entity_id as source1_entity_id,
                       g.{id_col} as {id_col}
                from read_parquet('{ART}/test_source1_normalized.parquet') a
                left join ({grouped_sql}) g on a.entity_id = g.s1_id
            ) to '{OUT / out_name}' (header, delimiter '\t')
        """
        duckdb.sql(query)
        n = duckdb.sql(f"select count(*) from read_csv('{OUT / out_name}', delim='\t', header=true)").fetchone()[0]
        print(f"{out_name}: {n} rows written", flush=True)

    build(tmp_cand, "candidate_pairs.tsv", "candidate_entity_ids")
    build(tmp_match, "matching_results.tsv", "matched_entity_ids")

    tmp_cand.unlink(missing_ok=True)
    tmp_match.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
