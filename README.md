Below are the full contents of the repository:

---

### README.md

````markdown
# OFI Feature Extraction

This repository provides a single self-contained Python script to compute Order Flow Imbalance (OFI) features from raw order-book data. The script reads a CSV of top-of-book events, calculates event-level flows, aggregates them into fixed time buckets, and exports the aggregated OFI metrics.

## File

- **Project.py**: Main script that:
  - Reads `first_25000_rows.csv` (detecting `ts_event` or `ts_recv` as the timestamp column)
  - Computes:
    - `best_ofi` (best-level imbalance)
    - `ofi_1` … `ofi_10` (level-specific imbalances)
    - `multi_level_ofi` (sum across levels)
    - `integrated_ofi` (PCA-based integration)
  - Writes the results to `ofi_features.csv` and prints a console preview

## Requirements

- Python 3.7 or higher
- pandas
- numpy
- scikit-learn

Install dependencies:
```bash
pip install pandas numpy scikit-learn
````

## Usage

1. Ensure `first_25000_rows.csv` is in the same directory as `Project.py`.
2. Run:

   ```bash
   python Project.py
   ```
3. The output file `ofi_features.csv` will contain:

   * `timestamp` (bucket start)
   * `best_ofi`
   * `ofi_1` … `ofi_10`
   * `multi_level_ofi`
   * `integrated_ofi`


