from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "machine-failure-prediction.csv"

if not RAW_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
required = [
    "UDI", "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear", "Failure",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

df = df.drop(columns=["UDI"])
X = df.drop(columns=["Failure"])
y = df["Failure"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# The workflow runs this script from the repository root.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_frame(name="Failure").to_csv("ytrain.csv", index=False)
ytest.to_frame(name="Failure").to_csv("ytest.csv", index=False)

print("Data prepared. Train rows:", len(Xtrain), "Test rows:", len(Xtest))
print("Type values:", sorted(X["Type"].dropna().unique()))
