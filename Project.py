import pandas as pd
import numpy as np

# --- Constants ---
DEPTH_LEVELS = 10

# --- Load and Prepare Data ---
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts_event')
    df = df.set_index('ts_event')
    return df

# --- Best-Level OFI ---
def compute_best_level_ofi(df: pd.DataFrame) -> pd.Series:
    bid_px, ask_px = df['bid_px_00'], df['ask_px_00']
    bid_sz, ask_sz = df['bid_sz_00'], df['ask_sz_00']

    bid_px_diff = bid_px.diff().fillna(0)
    ask_px_diff = ask_px.diff().fillna(0)
    bid_sz_diff = bid_sz.diff().fillna(0)
    ask_sz_diff = ask_sz.diff().fillna(0)

    bid_ofi = (bid_px_diff > 0) * bid_sz + (bid_px_diff == 0) * bid_sz_diff
    ask_ofi = (ask_px_diff < 0) * ask_sz + (ask_px_diff == 0) * (-ask_sz_diff)

    return (bid_ofi - ask_ofi).fillna(0)

# --- Multi-Level OFI ---
def compute_multilevel_ofi(df: pd.DataFrame, levels: int = 10) -> pd.Series:
    ofi = np.zeros(len(df))
    for i in range(levels):
        bid_px_diff = df[f'bid_px_{i:02}'].diff().fillna(0)
        ask_px_diff = df[f'ask_px_{i:02}'].diff().fillna(0)
        bid_sz_diff = df[f'bid_sz_{i:02}'].diff().fillna(0)
        ask_sz_diff = df[f'ask_sz_{i:02}'].diff().fillna(0)

        bid_ofi = (bid_px_diff > 0) * df[f'bid_sz_{i:02}'] + (bid_px_diff == 0) * bid_sz_diff
        ask_ofi = (ask_px_diff < 0) * df[f'ask_sz_{i:02}'] + (ask_px_diff == 0) * (-ask_sz_diff)

        ofi += (bid_ofi - ask_ofi).fillna(0)

    return pd.Series(ofi, index=df.index)

# --- Integrated OFI ---
def compute_integrated_ofi(ofi_series: pd.Series, window: str = '1s') -> pd.Series:
    return ofi_series.rolling(window=window).sum().fillna(0)

# --- Simulated Cross-Asset OFI ---
def simulate_cross_asset_ofi(base_ofi: pd.Series, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return base_ofi.shift(1).fillna(0) + np.random.normal(0, 1, len(base_ofi))

# --- Main Execution ---
def generate_ofi_features(csv_path: str) -> pd.DataFrame:
    df = load_data(csv_path)

    df['ofi_best_level'] = compute_best_level_ofi(df)
    df['ofi_multilevel'] = compute_multilevel_ofi(df, levels=DEPTH_LEVELS)
    df['ofi_integrated'] = compute_integrated_ofi(df['ofi_multilevel'], window='1s')
    df['cross_asset_ofi_msft_sim'] = simulate_cross_asset_ofi(df['ofi_multilevel'])

    return df[['ofi_best_level', 'ofi_multilevel', 'ofi_integrated', 'cross_asset_ofi_msft_sim']]

# --- Example Usage ---
if __name__ == "__main__":
    features_df = generate_ofi_features("first_25000_rows.csv")
    print(features_df.head())
