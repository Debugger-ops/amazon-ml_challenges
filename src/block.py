import duckdb, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "normalized"
OUT = ROOT / "artifacts" / "blocking"
PARTS = OUT / "_country_parts"
OUT.mkdir(parents=True, exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)
(OUT / "tmp").mkdir(exist_ok=True)

MAX_NAME_TOKEN_FREQ = 500
MAX_ADDR_TOKEN_FREQ = 500

def get_countries(con, split):
    countries = set()
    for src in ["source1", "source2", "source3"]:
        rows = con.execute(f"select distinct country from read_parquet('{ART}/{split}_{src}_normalized.parquet')").fetchall()
        countries.update(r[0] for r in rows)
    return sorted(countries)

def block_country(con, split, country, tbl):
    safe = country.replace(" ", "_").replace("/", "_")
    esc = country.replace("'", "''")
    out_path = PARTS / f"{split}_{tbl}_{safe}.parquet"
    q = f"""
    copy (
      with s1c as (
        select entity_id as s1_id, name_tokens, addr_tokens
        from read_parquet('{ART}/{split}_source1_normalized.parquet') where country = '{esc}'
      ),
      oc as (
        select entity_id as o_id, name_tokens, addr_tokens
        from read_parquet('{ART}/{split}_{tbl}_normalized.parquet') where country = '{esc}'
      ),

      s1_ntok as (select s1_id, unnest(name_tokens) as tok from s1c),
      o_ntok  as (select o_id,  unnest(name_tokens) as tok from oc),
      ntok_freq as (select tok, count(*) as freq from (select tok from s1_ntok union all select tok from o_ntok) group by tok),
      good_ntok as (select tok from ntok_freq where freq <= {MAX_NAME_TOKEN_FREQ}),
      name_pairs as (
        select distinct s1_id, o_id
        from (select s1_id, tok from s1_ntok semi join good_ntok using(tok)) s1f
        join (select o_id, tok from o_ntok semi join good_ntok using(tok)) of using(tok)
      ),

      s1_atok as (select s1_id, unnest(addr_tokens) as tok from s1c),
      o_atok  as (select o_id,  unnest(addr_tokens) as tok from oc),
      atok_freq as (select tok, count(*) as freq from (select tok from s1_atok union all select tok from o_atok) group by tok),
      good_atok as (select tok from atok_freq where freq <= {MAX_ADDR_TOKEN_FREQ}),
      addr_pairs as (
        select distinct s1_id, o_id
        from (select s1_id, tok from s1_atok semi join good_atok using(tok)) s1f
        join (select o_id, tok from o_atok semi join good_atok using(tok)) of using(tok)
      ),

      unioned as (
        select s1_id, o_id, 1 as by_name_token, 0 as by_addr_token from name_pairs
        union all
        select s1_id, o_id, 0, 1 from addr_pairs
      )
      select s1_id, o_id,
             max(by_name_token) as by_name_token,
             max(by_addr_token) as by_addr_token
      from unioned
      group by s1_id, o_id
    ) to '{out_path}' (format parquet)
    """
    con.execute(q)

def block(split: str):
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{OUT}/tmp'")

    countries = get_countries(con, split)
    print(f"countries found: {countries}", flush=True)

    for tbl in ["source2", "source3"]:
        for country in countries:
            print(f"blocking {split} vs {tbl} / country={country} ...", flush=True)
            block_country(con, split, country, tbl)

    for tbl in ["source2", "source3"]:
        out_path = OUT / f"{split}_{tbl}_pairs.parquet"
        con.execute(f"copy (select * from read_parquet('{PARTS}/{split}_{tbl}_*.parquet')) to '{out_path}' (format parquet)")
        print(f"combined -> {out_path.name}", flush=True)

    print("DONE", flush=True)

if __name__ == "__main__":
    block(sys.argv[1])