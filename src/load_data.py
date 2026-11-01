import pandas as pd
from pymongo import MongoClient
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=5000)

try:
    client.admin.command("ping")
    print("MongoDB connected successfully!")

    db = client["wine_quality_db"]

    
    datasets = {
        "red_wine": DATA_DIR / "winequality-red.csv",
        "white_wine": DATA_DIR / "winequality-white.csv",
    }

    for collection_name, file_path in datasets.items():
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path, sep=";")

   
        if "quality" not in df.columns:
            raise ValueError(
                f"'quality' column not found in {file_path.name}. "
                f"Detected columns: {df.columns.tolist()}"
            )

        records = df.to_dict(orient="records")
        collection = db[collection_name]

        
        collection.delete_many({})

        if records:
            collection.insert_many(records)

        print(f"\nDataset: {collection_name}")
        print(f"Rows loaded: {len(records)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"MongoDB documents: {collection.count_documents({})}")

    print("\nDataset import completed.")

except Exception as error:
    print(f"Error: {error}")

finally:
    client.close()