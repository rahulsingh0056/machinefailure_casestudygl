from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "machine-failure-prediction.csv"

if not RAW_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
expected_columns = [
    "UDI", "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear", "Failure",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Failure distribution:")
print(df["Failure"].value_counts(dropna=False))
