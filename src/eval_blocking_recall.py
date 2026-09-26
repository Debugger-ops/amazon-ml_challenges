import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GT = ROOT / "dataset" / "train" / "train_ground_truth.tsv"
BLOCK = ROOT / "artifacts" / "blocking"

con = duckdb.connect()
con.execute("SET threads=4")

con.execute(f"""
create temp table gt as
select source1_entity_id as s1_id, trim(unnest(str_split(matched_entity_ids, ','))) as o_id
from read_csv('{GT}', delim='\t', header=true)
where matched_entity_ids != ''
""")

for src in ['source2', 'source3']:
    prefix = 'S2-' if src == 'source2' else 'S3-'
    total = con.execute(f"select count(*) from gt where o_id like '{prefix}%'").fetchone()[0]
    uncapped = con.execute(f"""
        select count(*) from gt g
        join read_parquet('{BLOCK}/train_{src}_pairs.parquet') b
        on g.s1_id = b.s1_id and g.o_id = b.o_id
        where g.o_id like '{prefix}%'
    """).fetchone()[0]
    capped = con.execute(f"""
        select count(*) from gt g
        join read_parquet('{BLOCK}/train_{src}_pairs_capped.parquet') b
        on g.s1_id = b.s1_id and g.o_id = b.o_id
        where g.o_id like '{prefix}%'
    """).fetchone()[0]
    print(f"{src}: true={total} in_blocking={uncapped} ({uncapped/total*100:.2f}%) in_capped={capped} ({capped/total*100:.2f}%)")
