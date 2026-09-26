import duckdb, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import fbeta_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
FEAT = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["source","by_name_token","by_addr_token","name_jw_full","name_jw_core",
            "name_lev_sim","name_tok_jaccard","addr_jw_full","addr_lev_sim",
            "addr_tok_jaccard","pin_match","legal_marker_match","name_len_diff"]

NEG_KEEP_FRAC = 0.1

def build_splits():
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
    copy (
      select * from read_parquet('{FEAT}/train_features_labeled.parquet')
      where hash(s1_id) % 100 < 85 and (label = 1 or random() < {NEG_KEEP_FRAC})
    ) to '{FEAT}/train_sample.parquet' (format parquet)
    """)
    con.execute(f"""
    copy (
      select * from read_parquet('{FEAT}/train_features_labeled.parquet')
      where hash(s1_id) % 100 >= 85
    ) to '{FEAT}/val_full.parquet' (format parquet)
    """)
    print("splits written", flush=True)

def load(path):
    df = pd.read_parquet(path)
    X = df[FEATURES].astype("float32")
    y = df["label"].astype("int32")
    return df, X, y

def main():
    build_splits()
    _, X_train, y_train = load(FEAT / "train_sample.parquet")
    val_df, X_val, y_val = load(FEAT / "val_full.parquet")

    print(f"train rows: {len(X_train)}  positives: {y_train.sum()}", flush=True)
    print(f"val rows: {len(X_val)}  positives: {y_val.sum()}", flush=True)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    model = lgb.train(
        params, train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    val_proba = model.predict(X_val, num_iteration=model.best_iteration)

    best_t, best_f05 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (val_proba >= t).astype(int)
        f05 = fbeta_score(y_val, preds, beta=0.5, average="macro")
        if f05 > best_f05:
            best_f05, best_t = f05, t

    preds = (val_proba >= best_t).astype(int)
    print(f"BEST THRESHOLD: {best_t:.2f}")
    print(f"MACRO F0.5: {best_f05:.4f}")
    print(f"precision(1): {precision_score(y_val, preds):.4f}  recall(1): {recall_score(y_val, preds):.4f}")

    model.save_model(str(MODEL_DIR / "lgbm_model.txt"))
    with open(MODEL_DIR / "threshold.txt", "w") as f:
        f.write(str(best_t))
    print("model + threshold saved", flush=True)

if __name__ == "__main__":
    main()