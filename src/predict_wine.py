from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

client = MongoClient(
    "mongodb://127.0.0.1:27017/",
    serverSelectionTimeoutMS=5000
)

db = client["wine_quality_db"]

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def get_float(label):
    while True:
        try:
            value = float(input(f"{label}: "))
            if not pd.notna(value):
                raise ValueError
            return value
        except ValueError:
            print("Please enter a valid number.")


try:
    client.admin.command("ping")
    print("Connected to MongoDB.")

    print("\nWINE QUALITY PREDICTION")
    print("Choose a dataset/model:")
    print("1. Red wine")
    print("2. White wine")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice not in {"1", "2"}:
        raise ValueError("Please choose 1 or 2.")

    dataset = "red_wine" if choice == "1" else "white_wine"

    regressor_path = MODELS_DIR / f"{dataset}_regressor.joblib"
    classifier_path = MODELS_DIR / f"{dataset}_classifier.joblib"

    regressor = joblib.load(regressor_path)
    classifier = joblib.load(classifier_path)

    print("\nEnter the wine's chemical properties:")

    values = {feature: get_float(feature) for feature in FEATURES}
    input_df = pd.DataFrame([values], columns=FEATURES)

    predicted_quality = float(regressor.predict(input_df)[0])
    predicted_class = int(classifier.predict(input_df)[0])

    quality_label = (
        "Good quality (predicted class)"
        if predicted_class == 1
        else "Not good quality (predicted class)"
    )

    print("\n--- PREDICTION RESULT ---")
    print(f"Wine type: {dataset.replace('_', ' ').title()}")
    print(f"Predicted numerical quality: {predicted_quality:.2f}")
    print(f"Predicted category: {quality_label}")

    record = {
        "dataset": dataset,
        "input_features": values,
        "predicted_quality": round(predicted_quality, 2),
        "predicted_class": predicted_class,
        "predicted_category": (
            "Good" if predicted_class == 1 else "Not Good"
        ),
        "classification_rule": "quality >= 7 means Good",
        "created_at": datetime.now(timezone.utc),
    }

    result = db["predictions"].insert_one(record)

    print(f"\nPrediction saved to MongoDB.")
    print(f"Prediction document ID: {result.inserted_id}")

except Exception as error:
    print(f"\nError: {error}")

finally:
    client.close()
