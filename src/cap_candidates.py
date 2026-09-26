import duckdb, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "normalized"
BLOCK = ROOT / "artifacts" / "blocking"
PARTS = BLOCK / "_country_parts"
CAPPED = BLOCK / "_country_parts_capped"
CAPPED.mkdir(parents=True, exist_ok=True)
(BLOCK / "tmp").mkdir(exist_ok=True)

TOP_K = 40
NUM_SHARDS = 4

def cap_file(con, path, out_path, split, tbl):
    parts = []
    for shard in range(NUM_SHARDS):
        part_path = str(out_path) + f".shard{shard}.parquet"
        parts.append(part_path)
        q = f"""
        copy (
          with base as (
            select s1_id, o_id, by_name_token, by_addr_token
            from read_parquet('{path}')
            where hash(s1_id) % {NUM_SHARDS} = {shard}
          ),
          s1c as (
            select entity_id as s1_id, name_tokens, addr_tokens
            from read_parquet('{ART}/{split}_source1_normalized.parquet')
          ),
          oc as (
            select entity_id as o_id, name_tokens, addr_tokens
            from read_parquet('{ART}/{split}_{tbl}_normalized.parquet')
          ),
          scored as (
            select b.s1_id, b.o_id, b.by_name_token, b.by_addr_token,
              len(list_intersect(s1c.name_tokens, oc.name_tokens)) as name_ov,
              len(list_intersect(s1c.addr_tokens, oc.addr_tokens)) as addr_ov
            from base b
            join s1c using(s1_id)
            join oc using(o_id)
          ),
          ranked as (
            select *,
              row_number() over (
                partition by s1_id
                order by (by_name_token + by_addr_token) desc,
                         (name_ov + addr_ov) desc,
                         o_id
              ) as rn
            from scored
          )
          select s1_id, o_id, by_name_token, by_addr_token
          from ranked
          where rn <= {TOP_K}
        ) to '{part_path}' (format parquet)
        """
        con.execute(q)
    con.execute(f"copy (select * from read_parquet({parts})) to '{out_path}' (format parquet)")
    for p in parts:
        try:
            Path(p).unlink()
        except PermissionError:
            pass

def cap(split, tbl):
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{BLOCK}/tmp'")
    for f in sorted(PARTS.glob(f"{split}_{tbl}_*.parquet")):
        print(f"capping {f.name}", flush=True)
        cap_file(con, f, CAPPED / f.name, split, tbl)
    final_out = BLOCK / f"{split}_{tbl}_pairs_capped.parquet"
    con.execute(f"copy (select * from read_parquet('{CAPPED}/{split}_{tbl}_*.parquet')) to '{final_out}' (format parquet)")
    print(f"{split} {tbl} -> {final_out.name}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        cap(sys.argv[1], sys.argv[2])
    else:
        for split in ["train", "test"]:
            for tbl in ["source2", "source3"]:
                cap(split, tbl)