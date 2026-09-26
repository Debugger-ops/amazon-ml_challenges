# Business Entity Resolution — Pipeline

Matches noisy Source-2 / Source-3 business records against deduplicated Source-1
reference entities. Blocking + gradient-boosted classifier, threshold tuned
directly against the competition's entity-level macro F0.5 metric.

## Pipeline stages (run in order, from this directory)

```bash
pip install -r requirements.txt

python src/normalize.py train
python src/normalize.py test
python src/block.py train
python src/block.py test
python src/cap_candidates.py train
python src/cap_candidates.py test
python src/features.py train
python src/features.py test
python src/train.py
python src/eval_entity_f05.py     # scores validation split, picks best threshold
python src/predict.py             # writes output/candidate_pairs.tsv and matching_results.tsv
```

Each script resolves paths relative to the repo root (`student_resource/`), three
levels above `src/`, so all artifacts land under `student_resource/artifacts/`
and final outputs under `student_resource/output/`.

## Stage details

- **normalize.py** — cleans and tokenizes `business_name` / `business_address`
  per record: lowercasing, legal-suffix extraction (LLC/Ltd/Pvt etc.), name and
  address tokenization, PIN/postal-code candidate extraction, country field.
- **block.py** — candidate generation. For each country, joins Source-1 against
  Source-2/3 on any shared normalized name token OR address token, with a
  per-token frequency cap (`MAX_NAME_TOKEN_FREQ` / `MAX_ADDR_TOKEN_FREQ`,
  default 500) to bound compute cost on extremely common words.
- **cap_candidates.py** — caps the number of candidate pairs per Source-1
  entity to keep downstream feature computation and training tractable.
- **features.py** — computes pairwise similarity features (Jaro-Winkler,
  Levenshtein, token Jaccard on both name and address, PIN match, legal-marker
  match, name length difference, blocking-source flags).
- **train.py** — trains a LightGBM binary classifier on labeled candidate
  pairs (label = true match per `train_ground_truth.tsv`), with an
  entity-level train/validation split.
- **eval_entity_f05.py** — scores the validation split and sweeps
  classification thresholds directly against entity-level macro F0.5 (the
  actual competition metric, including singleton credit), not pairwise F0.5.
  Writes the chosen threshold to `artifacts/model/threshold.txt`.
- **predict.py** — scores test candidates with the trained model and the
  tuned threshold, writes `output/candidate_pairs.tsv` and
  `output/matching_results.tsv`.

## Known limitation

Blocking recall is ~80% (measured directly against `train_ground_truth.tsv`):
about 20% of true matches share zero name/address token with their Source-1
counterpart due to heavy typos, transliteration, or corruption in Source-2/3,
and are never seen by the classifier. Raising the token-frequency cap does not
fix this cheaply, since the highest-value words (e.g. "group", "ventures") are
common enough that including them would blow up candidate volume combinatorially.
Fixing this fully would need a different blocking strategy (e.g. sorted-neighborhood
or n-gram-based blocking) — noted here as a direction for future work.
