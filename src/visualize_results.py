from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

client = MongoClient(
    "mongodb://127.0.0.1:27017/",
    serverSelectionTimeoutMS=5000
)

try:
    db = client["wine_quality_db"]

    records = list(
        db["model_metrics"].find(
            {},
            {"_id": 0}
        )
    )

    if not records:
        raise ValueError(
            "No model metrics found. Run train_models.py first."
        )

    
    latest = {}

    for record in records:
        name = record["dataset"]
        timestamp = record.get("created_at")

        if (
            name not in latest
            or timestamp > latest[name].get("created_at")
        ):
            latest[name] = record

    datasets = list(latest.keys())

    regression_rows = []
    classification_rows = []

    for name in datasets:
        result = latest[name]

        regression_rows.append({
            "Dataset": name.replace("_", " ").title(),
            "R² Score": result["regression"]["r2_score"],
            "MAE": result["regression"]["mae"],
        })

        classification_rows.append({
            "Dataset": name.replace("_", " ").title(),
            "Accuracy": result["classification"]["accuracy"],
            "Precision": result["classification"]["precision"],
            "Recall": result["classification"]["recall"],
            "F1-score": result["classification"]["f1_score"],
        })

    regression_df = pd.DataFrame(regression_rows)
    classification_df = pd.DataFrame(classification_rows)

    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sns.barplot(
        data=regression_df,
        x="Dataset",
        y="R² Score",
        ax=axes[0]
    )
    axes[0].set_title("Regression: R² Score")
    axes[0].set_ylim(0, 1)

    sns.barplot(
        data=regression_df,
        x="Dataset",
        y="MAE",
        ax=axes[1]
    )
    axes[1].set_title("Regression: Mean Absolute Error")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "regression_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    
    classification_long = classification_df.melt(
        id_vars="Dataset",
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=classification_long,
        x="Metric",
        y="Score",
        hue="Dataset"
    )

    plt.title("Random Forest Classification Performance")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "classification_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 4))

    if len(datasets) == 1:
        axes = [axes]

    for ax, name in zip(axes, datasets):
        matrix = latest[name]["classification"]["confusion_matrix"]

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Good", "Good"],
            yticklabels=["Not Good", "Good"],
            ax=ax
        )

        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    
    regression_df.to_csv(
        OUTPUT_DIR / "regression_metrics.csv",
        index=False
    )

    classification_df.to_csv(
        OUTPUT_DIR / "classification_metrics.csv",
        index=False
    )

    print("Visualizations created successfully!")
    print("Output folder:", OUTPUT_DIR)
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(file.name)

finally:
    client.close()
