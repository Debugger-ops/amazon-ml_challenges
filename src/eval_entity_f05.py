import duckdb, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FEAT = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "model"
ART = ROOT / "artifacts" / "normalized"

FEATURES = ["source","by_name_token","by_addr_token","name_jw_full","name_jw_core",
            "name_lev_sim","name_tok_jaccard","addr_jw_full","addr_lev_sim",
            "addr_tok_jaccard","pin_match","legal_marker_match","name_len_diff"]

def score_val():
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
    df = pd.read_parquet(FEAT / "val_full.parquet")
    X = df[FEATURES].astype("float32")
    df["proba"] = model.predict(X)
    df[["s1_id", "label", "proba"]].to_parquet(MODEL_DIR / "val_scored.parquet")
    print(f"scored {len(df)} val rows", flush=True)

def sweep():
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95,
                  0.96, 0.97, 0.98, 0.99]
    best_t, best_f05 = None, -1
    for t in thresholds:
        q = f"""
        with scored as (
          select s1_id, label, proba from read_parquet('{MODEL_DIR}/val_scored.parquet')
        ),
        agg as (
          select s1_id,
                 sum(case when proba >= {t} then 1 else 0 end) as pred_n,
                 sum(case when proba >= {t} and label = 1 then 1 else 0 end) as tp,
                 sum(label) as true_n
          from scored
          group by s1_id
        ),
        full_ids as (
          select entity_id as s1_id
          from read_parquet('{ART}/train_source1_normalized.parquet')
          where hash(entity_id) % 100 >= 85
        ),
        joined as (
          select f.s1_id, coalesce(a.pred_n,0) as pred_n, coalesce(a.tp,0) as tp, coalesce(a.true_n,0) as true_n
          from full_ids f left join agg a using(s1_id)
        ),
        scored_f as (
          select
            case
              when true_n = 0 and pred_n = 0 then 1.0
              when true_n = 0 and pred_n > 0 then 0.0
              when pred_n = 0 and true_n > 0 then 0.0
                when tp = 0 then 0.0
              else (1.25 * (tp::double/pred_n) * (tp::double/true_n)) / (0.25*(tp::double/pred_n) + (tp::double/true_n))
            end as f05
          from joined
        )
        select avg(f05) from scored_f
        """
        f05 = con.execute(q).fetchone()[0]
        print(f"threshold={t:.2f}  entity-macro-F0.5={f05:.4f}", flush=True)
        if f05 > best_f05:
            best_f05, best_t = f05, t
    print(f"\nBEST: threshold={best_t}  entity-macro-F0.5={best_f05:.4f}")
    with open(MODEL_DIR / "threshold.txt", "w") as f:
        f.write(str(best_t))
    print("threshold.txt updated", flush=True)

if __name__ == "__main__":
    score_val()
    sweep()