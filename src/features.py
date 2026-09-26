import duckdb, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "normalized"
BLOCK = ROOT / "artifacts" / "blocking"
CAPPED = BLOCK / "_country_parts_capped"
DATASET = ROOT / "dataset"
FEAT = ROOT / "artifacts" / "features"
FPARTS = FEAT / "_parts"
FPARTS.mkdir(parents=True, exist_ok=True)
(FEAT / "tmp").mkdir(exist_ok=True)

def featurize_file(con, path, out_path, split, tbl, country):
    esc = country.replace("'", "''")
    src_num = 2 if tbl == "source2" else 3
    q = f"""
    copy (
      with pairs as (
        select s1_id, o_id, by_name_token, by_addr_token
        from read_parquet('{path}')
      ),
      s1c as (
        select entity_id as s1_id,
               name_normalized_full as s1_name_full,
               name_normalized_core as s1_name_core,
               name_tokens as s1_name_tok,
               addr_normalized_full as s1_addr_full,
               addr_tokens as s1_addr_tok,
               addr_pin_candidates as s1_pins,
               name_legal_markers as s1_legal
        from read_parquet('{ART}/{split}_source1_normalized.parquet')
        where country = '{esc}'
      ),
      oc as (
        select entity_id as o_id,
               name_normalized_full as o_name_full,
               name_normalized_core as o_name_core,
               name_tokens as o_name_tok,
               addr_normalized_full as o_addr_full,
               addr_tokens as o_addr_tok,
               addr_pin_candidates as o_pins,
               name_legal_markers as o_legal
        from read_parquet('{ART}/{split}_{tbl}_normalized.parquet')
        where country = '{esc}'
      )
      select
        p.s1_id, p.o_id, {src_num} as source,
        p.by_name_token, p.by_addr_token,
        jaro_winkler_similarity(s1c.s1_name_full, oc.o_name_full) as name_jw_full,
        jaro_winkler_similarity(s1c.s1_name_core, oc.o_name_core) as name_jw_core,
        1.0 - levenshtein(s1c.s1_name_core, oc.o_name_core)::double /
          greatest(length(s1c.s1_name_core), length(oc.o_name_core), 1) as name_lev_sim,
        len(list_intersect(s1c.s1_name_tok, oc.o_name_tok))::double /
          greatest(len(list_distinct(list_concat(s1c.s1_name_tok, oc.o_name_tok))), 1) as name_tok_jaccard,
        jaro_winkler_similarity(s1c.s1_addr_full, oc.o_addr_full) as addr_jw_full,
        1.0 - levenshtein(s1c.s1_addr_full, oc.o_addr_full)::double /
          greatest(length(s1c.s1_addr_full), length(oc.o_addr_full), 1) as addr_lev_sim,
        len(list_intersect(s1c.s1_addr_tok, oc.o_addr_tok))::double /
          greatest(len(list_distinct(list_concat(s1c.s1_addr_tok, oc.o_addr_tok))), 1) as addr_tok_jaccard,
        case when len(list_intersect(s1c.s1_pins, oc.o_pins)) > 0 then 1 else 0 end as pin_match,
        case when len(list_intersect(s1c.s1_legal, oc.o_legal)) > 0 then 1 else 0 end as legal_marker_match,
        abs(length(s1c.s1_name_core) - length(oc.o_name_core)) as name_len_diff
      from pairs p
      join s1c using(s1_id)
      join oc using(o_id)
    ) to '{out_path}' (format parquet)
    """
    con.execute(q)

def featurize(split, tbl):
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{FEAT}/tmp'")
    prefix = f"{split}_{tbl}_"
    for f in sorted(CAPPED.glob(f"{split}_{tbl}_*.parquet")):
        country = f.stem[len(prefix):].replace("_", " ")
        print(f"featurizing {f.name}", flush=True)
        featurize_file(con, f, FPARTS / f.name, split, tbl, country)
    out = FEAT / f"{split}_{tbl}_features.parquet"
    con.execute(f"copy (select * from read_parquet('{FPARTS}/{split}_{tbl}_*.parquet')) to '{out}' (format parquet)")
    print(f"-> {out.name}", flush=True)

def combine_and_label(split):
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    if split == "train":
        con.execute(f"""
        create view gt as
        select source1_entity_id as s1_id, trim(unnest(string_split(matched_entity_ids, ','))) as match_id
        from read_csv('{DATASET}/train/train_ground_truth.tsv', delim='\t', header=true, quote='')
        where matched_entity_ids is not null and matched_entity_ids != ''
        """)
        q = f"""
        copy (
          select f.*, case when g.match_id is not null then 1 else 0 end as label
          from read_parquet('{FEAT}/train_source2_features.parquet') f
          left join gt g on g.s1_id = f.s1_id and g.match_id = f.o_id
          union all
          select f.*, case when g.match_id is not null then 1 else 0 end as label
          from read_parquet('{FEAT}/train_source3_features.parquet') f
          left join gt g on g.s1_id = f.s1_id and g.match_id = f.o_id
        ) to '{FEAT}/train_features_labeled.parquet' (format parquet)
        """
    else:
        q = f"""
        copy (
          select * from read_parquet('{FEAT}/test_source2_features.parquet')
          union all
          select * from read_parquet('{FEAT}/test_source3_features.parquet')
        ) to '{FEAT}/test_features_final.parquet' (format parquet)
        """
    con.execute(q)
    print(f"{split} combined", flush=True)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        featurize(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        combine_and_label("train" if sys.argv[1] == "combine_train" else "test")