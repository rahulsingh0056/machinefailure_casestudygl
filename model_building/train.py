from pathlib import Path
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT_DIR / "deployment"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric_features = [
    "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear",
]
categorical_features = ["Type"]

counts = ytrain.value_counts()
if 0 not in counts.index or 1 not in counts.index:
    raise ValueError(f"Expected binary target labels 0 and 1, found: {counts.to_dict()}")
class_weight = counts[0] / counts[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
)
pipeline = make_pipeline(preprocessor, model)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1, error_score="raise"
)
grid.fit(Xtrain, ytrain)

best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print(classification_report(ytest, best_model.predict(Xtest), zero_division=0))

model_path = DEPLOY_DIR / "best_machine_failure_model_v1.joblib"
joblib.dump(best_model, model_path)
print("Model saved to:", model_path)
