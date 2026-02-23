"""Merge & sort the UAVDT 50-sequence results into a single clean CSV."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / "uavdt_results.csv"
OUT_CSV = ROOT / "uavdt_all50_results.csv"

EXPECTED_COLS = [
    "sequence", "frame_id", "model", "tp", "fp", "fn",
    "recall", "precision", "f1", "mean_iou",
    "weather", "altitude", "camera_view",
    "num_gt", "num_gt_occ1", "num_gt_occ2", "num_gt_occ3", "num_gt_occ4",
    "num_gt_outofview",
]


def main():
    df = pd.read_csv(IN_CSV)
    assert list(df.columns) == EXPECTED_COLS, f"Column mismatch: {list(df.columns)}"

    df_sorted = df.sort_values(["sequence", "frame_id", "model"]).reset_index(drop=True)
    df_sorted.to_csv(OUT_CSV, index=False)

    # Summary
    print(f"Total rows:      {len(df_sorted)}", flush=True)
    print(f"Total sequences: {df_sorted['sequence'].nunique()}", flush=True)

    print("\n=== Per-Weather Breakdown ===", flush=True)
    weather = df_sorted.groupby("weather").agg(
        sequences=("sequence", "nunique"),
        rows=("sequence", "count"),
    )
    print(weather.to_string(), flush=True)

    print("\n=== Mean Recall per Model ===", flush=True)
    recall = df_sorted.groupby("model")["recall"].mean().round(4)
    print(recall.to_string(), flush=True)

    print(f"\nSaved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
