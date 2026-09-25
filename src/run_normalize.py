import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_name, normalize_address

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts" / "normalized"
PARTS_DIR = ARTIFACTS_DIR / "_parts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)

NAME_DTYPE = pl.Struct([
    pl.Field("normalized_full", pl.Utf8),
    pl.Field("normalized_core", pl.Utf8),
    pl.Field("tokens", pl.List(pl.Utf8)),
    pl.Field("has_devanagari", pl.Boolean),
    pl.Field("legal_markers", pl.List(pl.Utf8)),
])
ADDR_DTYPE = pl.Struct([
    pl.Field("normalized_full", pl.Utf8),
    pl.Field("tokens", pl.List(pl.Utf8)),
    pl.Field("pin_candidates", pl.List(pl.Utf8)),
])

def transform(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        pl.col("business_name").map_elements(normalize_name, return_dtype=NAME_DTYPE).alias("_name")
    ).with_columns(
        pl.col("_name").struct.rename_fields([
            "name_normalized_full", "name_normalized_core", "name_tokens",
            "name_has_devanagari", "name_legal_markers",
        ])
    ).unnest("_name")

    lf = lf.with_columns(
        pl.col("business_address").map_elements(normalize_address, return_dtype=ADDR_DTYPE).alias("_addr")
    ).with_columns(
        pl.col("_addr").struct.rename_fields([
            "addr_normalized_full", "addr_tokens", "addr_pin_candidates",
        ])
    ).unnest("_addr")
    return lf

def process_chunk(split: str, source: str, offset: int, length: int):
    in_path = DATASET_DIR / split / f"{split}_{source}.tsv"
    out_path = PARTS_DIR / f"{split}_{source}_part_{offset}.parquet"
    lf = pl.scan_csv(in_path, separator="\t").slice(offset, length)
    lf = transform(lf)
    lf.sink_parquet(out_path)
    n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    print(f"  chunk offset={offset} len={length} -> wrote {n} rows to {out_path.name}", flush=True)

def combine(split: str, source: str):
    parts = sorted(PARTS_DIR.glob(f"{split}_{source}_part_*.parquet"),
                    key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if not parts:
        print(f"  no parts found for {split}_{source}"); return
    out_path = ARTIFACTS_DIR / f"{split}_{source}_normalized.parquet"
    dfs = [pl.scan_parquet(p) for p in parts]
    pl.concat(dfs).sink_parquet(out_path)
    n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    print(f"  combined {len(parts)} parts -> {out_path.name} ({n} rows)", flush=True)
    for p in parts:
        try:
            p.unlink()
        except PermissionError:
            pass  # cleanup not permitted, harmless leftover

def process_whole(split: str, source: str):
    in_path = DATASET_DIR / split / f"{split}_{source}.tsv"
    out_path = ARTIFACTS_DIR / f"{split}_{source}_normalized.parquet"
    lf = transform(pl.scan_csv(in_path, separator="\t"))
    lf.sink_parquet(out_path)
    n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    print(f"  wrote {n} rows", flush=True)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "whole":
        process_whole(sys.argv[2], sys.argv[3])
    elif cmd == "chunk":
        process_chunk(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "combine":
        combine(sys.argv[2], sys.argv[3])
    print("DONE", flush=True)
