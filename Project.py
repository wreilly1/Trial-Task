#!/usr/bin/env python3

import os
import sys

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def compute_event_flows(df, max_level=10):
    """
    Computes event-level OF_m = OF_{m,b} - OF_{m,a} for m=0...(max_level-1)
    per Eq (1)-(2) in Cont et al. (2023).
    Expects columns: bid_px_{m:02d}, ask_px_{m:02d}, bid_sz_{m:02d}, ask_sz_{m:02d}.
    Returns df with added columns OF_0 ... OF_{max_level-1}.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    for m in range(max_level):
        bp = df[f'bid_px_{m:02d}']
        ap = df[f'ask_px_{m:02d}']
        bs = df[f'bid_sz_{m:02d}']
        asz = df[f'ask_sz_{m:02d}']

        bp_prev = bp.shift(1)
        bs_prev = bs.shift(1)
        ap_prev = ap.shift(1)
        asz_prev = asz.shift(1)

        # event‐level bid flow
        ofb = np.where(
            bp > bp_prev,
            bs,
            np.where(bp == bp_prev, bs - bs_prev, -bs)
        )
        # event‐level ask flow
        ofa = np.where(
            ap > ap_prev,
            -asz,
            np.where(ap == ap_prev, asz - asz_prev, asz)
        )

        # wrap in Series to use fillna
        ofi_m = pd.Series(ofb - ofa, index=df.index).fillna(0)
        df[f'OF_{m}'] = ofi_m

    return df


def aggregate_ofi(df, freq='1S', max_level=10):
    """
    Aggregates event-level OF_m into time buckets of width `freq`.
    Returns a DataFrame indexed by bucket-start timestamp with columns:
      - best_ofi        := sum OF_0         (Eq 1)
      - ofi_1 ... ofi_M := sum OF_m         (Eq 2)
      - multi_level_ofi := sum_m ofi_m
      - integrated_ofi  := PCA1(sum_m ofi_m) normalized by l1-norm (Eq 4)
    """
    # floor timestamps into buckets
    df['interval'] = df['timestamp'].dt.floor(freq)
    grp = df.groupby('interval')

    # best‐level OFI
    best = grp['OF_0'].sum().rename('best_ofi')

    # per‐level deeper OFI
    ofi_levels = pd.DataFrame({
        f'ofi_{m + 1}': grp[f'OF_{m}'].sum()
        for m in range(max_level)
    }, index=best.index)

    # multi‐level OFI = sum across levels
    ofi_levels['multi_level_ofi'] = ofi_levels.sum(axis=1)

    # integrated OFI via first PCA component
    X = ofi_levels[[f'ofi_{m + 1}' for m in range(max_level)]].values
    pca = PCA(n_components=1)
    pca.fit(X)
    w = pca.components_[0]
    w = w / np.sum(np.abs(w))  # ℓ1‐normalize
    ofi_levels['integrated_ofi'] = X.dot(w)

    # combine best‐level + deeper + integrated
    result = pd.concat([best, ofi_levels], axis=1)
    result.index.name = 'timestamp'
    return result


def compute_cross_asset_ofi(ofi_dict):
    """
    Given {symbol: DataFrame with 'best_ofi' column},
    returns {symbol: DataFrame with cross_ofi series = sum of other symbols' best_ofi}.
    """
    cross = {}
    for sym, df in ofi_dict.items():
        others = [ofi_dict[o]['best_ofi'] for o in ofi_dict if o != sym]
        cross_ofi = pd.Series(data=sum(others), index=df.index).rename('cross_ofi')
        result = pd.DataFrame(cross_ofi)
        cross[sym] = result
    return cross


def main():
    filename = 'first_25000_rows.csv'
    if not os.path.exists(filename):
        sys.exit(f"Error: '{filename}' not found in {os.getcwd()}")

    # 1) Load raw CSV
    df = pd.read_csv(filename)

    # 2) Detect & rename timestamp column
    for c in ('ts_event', 'ts_recv', 'timestamp', 'time', 'datetime'):
        if c in df.columns:
            df = df.rename(columns={c: 'timestamp'})
            try:
                # if integer nanoseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            except (ValueError, TypeError):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            break
    else:
        print("Columns found:", df.columns.tolist())
        sys.exit("Error: no timestamp-like column (ts_event/ts_recv/etc.) found.")

    # 3) Compute event‐level OFI
    df = compute_event_flows(df, max_level=10)

    # 4) Aggregate into 1-second buckets
    ofi_feats = aggregate_ofi(df, freq='1S', max_level=10)

    # 5) Save to CSV
    out_fn = 'ofi_features.csv'
    ofi_feats.to_csv(out_fn)

    # 6) Print summary
    print("✅ OFI extraction complete!")
    print(f"  • Wrote: {out_fn}")
    print(f"  • Shape: {ofi_feats.shape[0]} rows × {ofi_feats.shape[1]} cols")
    print("  • Preview:")
    print(ofi_feats.head().to_string())


if __name__ == '__main__':
    main()

