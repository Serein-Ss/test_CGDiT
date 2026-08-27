# Magndata-Tc benchmark dataset

This directory contains the fixed dataset used by all four Curie-temperature
regression arms.

- `source_magndata_tc.csv`: source table after the original equivalence/Tc
  deduplication (1297 rows).
- `all.csv`: standardized primitive, Niggli-reduced structures accepted for
  training (1284 rows).
- `train.csv`, `val.csv`, `test.csv`: formula-disjoint 8:1:1 splits containing
  1023, 122 and 139 rows.
- `rejected.csv`: 12 structures above the 100-atom primitive-cell limit and one
  structure whose symmetry reduction failed.
- `split_manifest.json`: hashes, target statistics, split policy and leakage
  checks.

The target column is `tc` in Kelvin. Formula groups never cross splits. The
training-only target mean and population standard deviation are recorded in
`conf/data/magndata.yaml`.

To reproduce the packaged splits:

```bash
python -m scripts.cli.data.prepare_magndata_tc \
  --source data/magndata/source_magndata_tc.csv \
  --output-dir data/magndata \
  --seed 42 \
  --max-atoms 100
```
