import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "wine_quality_db"
RANDOM_STATE = 42

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]


def train_for_dataset(dataset_name):
    print(f"\n{'=' * 55}")
    print(f"Training models for {dataset_name}")
    print("=" * 55)

    
    documents = list(db[dataset_name].find({}, {"_id": 0}))
    df = pd.DataFrame(documents)

    if df.empty:
        raise ValueError(f"No data found in MongoDB collection: {dataset_name}")

    if "quality" not in df.columns:
        raise ValueError("The dataset must contain a quality column.")

    X = df.drop(columns=["quality"])
    y = df["quality"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    
    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)

    reg_predictions = regressor.predict(X_test)

    regression_metrics = {
        "r2_score": round(float(r2_score(y_test, reg_predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, reg_predictions)), 4),
    }

    
    y_class = (y >= 7).astype(int)
    y_class_train = y_class.loc[X_train.index]
    y_class_test = y_class.loc[X_test.index]

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(X_train, y_class_train)

    class_predictions = classifier.predict(X_test)

    classification_metrics = {
        "accuracy": round(float(accuracy_score(y_class_test, class_predictions)), 4),
        "precision": round(
            float(precision_score(y_class_test, class_predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_class_test, class_predictions, zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_class_test, class_predictions, zero_division=0)), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_class_test, class_predictions, labels=[0, 1]
        ).tolist(),
    }

    joblib.dump(regressor, MODELS_DIR / f"{dataset_name}_regressor.joblib")
    joblib.dump(classifier, MODELS_DIR / f"{dataset_name}_classifier.joblib")

    
    result = {
        "dataset": dataset_name,
        "regression": regression_metrics,
        "classification": classification_metrics,
        "rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "classification_rule": "quality >= 7 means good quality (class 1)",
        "algorithm": "Random Forest",
        "random_state": RANDOM_STATE,
        "created_at": datetime.now(timezone.utc),
    }

    db["model_metrics"].insert_one(result)

    print("\nRegression metrics:")
    print(json.dumps(regression_metrics, indent=2))

    print("\nClassification metrics:")
    print(json.dumps(classification_metrics, indent=2))

    print("\nModels saved in:", MODELS_DIR)
    print("Metrics saved in MongoDB collection: model_metrics")


try:
    client.admin.command("ping")
    print("MongoDB connection successful.")

    for dataset in ["red_wine", "white_wine"]:
        train_for_dataset(dataset)

    print("\nAll model training and evaluation completed.")

except Exception as error:
    print(f"\nERROR: {error}")
    raise

finally:
    client.close()
