from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

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

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
)

# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_models():
    models = {}

    for wine_type in ["red_wine", "white_wine"]:
        models[wine_type] = {
            "regressor": joblib.load(
                MODELS_DIR / f"{wine_type}_regressor.joblib"
            ),
            "classifier": joblib.load(
                MODELS_DIR / f"{wine_type}_classifier.joblib"
            ),
        }

    return models


# -------------------- DATABASE CONNECTION --------------------

@st.cache_resource
def get_mongo_client():
    try:
        mongo_uri = st.secrets["MONGO_URI"]
    except (KeyError, FileNotFoundError):
        mongo_uri = "mongodb://127.0.0.1:27017/"

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
    )
    client.admin.command("ping")
    return client


# -------------------- PAGE HEADER --------------------

st.title("🍷 Wine Quality Prediction System")

st.markdown(
    """
    ### Machine Learning + MongoDB

    Predict wine quality using its chemical properties.
    Select the wine type, enter the measurements, and click **Predict Quality**.
    """
)

st.divider()

# -------------------- INPUT FORM --------------------

left, right = st.columns([1, 1])

with left:
    st.subheader("🧪 Wine Information")

    wine_choice = st.selectbox(
        "Select wine type",
        ["White Wine", "Red Wine"],
    )

    wine_type = (
        "white_wine" if wine_choice == "White Wine" else "red_wine"
    )

    st.caption("Enter the chemical properties below.")

    with st.form("wine_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            fixed_acidity = st.number_input(
                "Fixed acidity",
                min_value=0.0,
                value=6.8,
                step=0.1,
                format="%.4f",
            )

            volatile_acidity = st.number_input(
                "Volatile acidity",
                min_value=0.0,
                value=0.25,
                step=0.01,
                format="%.4f",
            )

            citric_acid = st.number_input(
                "Citric acid",
                min_value=0.0,
                value=0.30,
                step=0.01,
                format="%.4f",
            )

            residual_sugar = st.number_input(
                "Residual sugar",
                min_value=0.0,
                value=2.5,
                step=0.1,
                format="%.4f",
            )

            chlorides = st.number_input(
                "Chlorides",
                min_value=0.0,
                value=0.040,
                step=0.001,
                format="%.4f",
            )

            free_sulfur_dioxide = st.number_input(
                "Free sulfur dioxide",
                min_value=0.0,
                value=25.0,
                step=1.0,
                format="%.4f",
            )

        with col2:
            total_sulfur_dioxide = st.number_input(
                "Total sulfur dioxide",
                min_value=0.0,
                value=100.0,
                step=1.0,
                format="%.4f",
            )

            density = st.number_input(
                "Density",
                min_value=0.0,
                value=0.9915,
                step=0.0001,
                format="%.4f",
            )

            ph = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=3.20,
                step=0.01,
                format="%.4f",
            )

            sulphates = st.number_input(
                "Sulphates",
                min_value=0.0,
                value=0.50,
                step=0.01,
                format="%.4f",
            )

            alcohol = st.number_input(
                "Alcohol (%)",
                min_value=0.0,
                value=11.5,
                step=0.1,
                format="%.4f",
            )

        submitted = st.form_submit_button(
            "🔍 Predict Quality",
            use_container_width=True,
            type="primary",
        )

# -------------------- PREDICTION --------------------

with right:
    st.subheader("📊 Prediction Results")

    if submitted:
        input_values = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_sulfur_dioxide,
            "total sulfur dioxide": total_sulfur_dioxide,
            "density": density,
            "pH": ph,
            "sulphates": sulphates,
            "alcohol": alcohol,
        }

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURES,
        )

        try:
            models = load_models()
            selected_models = models[wine_type]

            predicted_quality = float(
                selected_models["regressor"].predict(input_df)[0]
            )

            predicted_class = int(
                selected_models["classifier"].predict(input_df)[0]
            )

            category = "Good" if predicted_class == 1 else "Not Good"

            st.success("Prediction completed!")

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Predicted Quality",
                f"{predicted_quality:.2f} / 10",
            )

            metric2.metric(
                "Predicted Category",
                category,
            )

            if predicted_class == 1:
                st.info(
                    "The classifier predicts that this wine is in "
                    "the good-quality category."
                )
            else:
                st.warning(
                    "The classifier predicts that this wine is not "
                    "in the good-quality category."
                )

            st.caption(
                "Classification rule: quality ≥ 7 means Good. "
                "The numerical score is a regression estimate; "
                "the category is a separate classification prediction."
            )

            # Save prediction to MongoDB
            try:
                client = get_mongo_client()
                db = client["wine_quality_db"]

                record = {
                    "dataset": wine_type,
                    "input_features": input_values,
                    "predicted_quality": round(predicted_quality, 2),
                    "predicted_class": predicted_class,
                    "predicted_category": category,
                    "classification_rule": "quality >= 7 means Good",
                    "created_at": datetime.now(timezone.utc),
                    "source": "streamlit_app",
                }

                result = db["predictions"].insert_one(record)

                st.success(
                    f"Prediction saved to MongoDB. ID: {result.inserted_id}"
                )

            except PyMongoError as error:
                st.warning(
                    "Prediction succeeded, but it could not be saved "
                    f"to MongoDB: {error}"
                )

        except FileNotFoundError:
            st.error(
                "A trained model file is missing. Check your models folder."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.info(
            "Enter the wine's chemical properties and click "
            "'Predict Quality' to see the results."
        )

st.divider()
# -------------------- PREDICTION HISTORY --------------------

st.header("📜 Prediction History")

try:
    client = get_mongo_client()
    db = client["wine_quality_db"]
    collection = db["predictions"]

    history = list(
        collection.find(
            {},
            {
                "_id": 0,
                "dataset": 1,
                "predicted_quality": 1,
                "predicted_category": 1,
                "created_at": 1,
                "source": 1,
            },
        ).sort("created_at", -1).limit(500)
    )

    if history:
        history_df = pd.DataFrame(history)

        history_df = history_df.rename(
            columns={
                "dataset": "Wine Type",
                "predicted_quality": "Predicted Quality",
                "predicted_category": "Category",
                "created_at": "Prediction Time",
                "source": "Source",
            }
        )

        if "Wine Type" in history_df.columns:
            history_df["Wine Type"] = (
                history_df["Wine Type"]
                .fillna("Unknown")
                .str.replace("_", " ", regex=False)
                .str.title()
            )
        else:
            history_df["Wine Type"] = "Unknown"

        if "Category" not in history_df.columns:
            history_df["Category"] = "Unknown"
        else:
            history_df["Category"] = history_df["Category"].fillna("Unknown")

        if "Predicted Quality" not in history_df.columns:
            history_df["Predicted Quality"] = None

        if "Prediction Time" not in history_df.columns:
            history_df["Prediction Time"] = None

        if "Source" not in history_df.columns:
            history_df["Source"] = "Unknown"

        # Filters
        filter1, filter2 = st.columns(2)

        with filter1:
            wine_filter = st.selectbox(
                "Filter by wine type",
                ["All", "Red Wine", "White Wine"],
            )

        with filter2:
            category_filter = st.selectbox(
                "Filter by predicted category",
                ["All", "Good", "Not Good"],
            )

        filtered_df = history_df.copy()

        if wine_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Wine Type"] == wine_filter
            ]

        if category_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Category"] == category_filter
            ]

        st.write(
            f"Showing {len(filtered_df)} matching predictions "
            f"out of {len(history_df)} retrieved records."
        )

        display_columns = [
            "Wine Type",
            "Predicted Quality",
            "Category",
            "Prediction Time",
            "Source",
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Matching Predictions", len(filtered_df))

        good_count = int(
            (filtered_df["Category"] == "Good").sum()
        )
        not_good_count = int(
            (filtered_df["Category"] == "Not Good").sum()
        )

        col2.metric("Good Predictions", good_count)
        col3.metric("Not Good Predictions", not_good_count)

        # Download filtered results
        csv_data = filtered_df[display_columns].to_csv(index=False)

        st.download_button(
            label="⬇️ Download Filtered History (CSV)",
            data=csv_data,
            file_name="wine_prediction_history.csv",
            mime="text/csv",
        )

    else:
        st.info("No predictions have been saved yet.")

except PyMongoError as error:
    st.error(f"Unable to retrieve prediction history: {error}")

# -------------------- MODEL PERFORMANCE --------------------

st.header("📈 Model Performance Dashboard")

st.write(
    "Performance metrics for the Random Forest regression and "
    "classification models, retrieved directly from MongoDB."
)

try:
    client = get_mongo_client()
    db = client["wine_quality_db"]

    metric_docs = list(
        db["model_metrics"].find({}).sort("created_at", -1)
    )

    # Keep the most recent evaluation document for each wine type
    latest_metrics = {}

    for doc in metric_docs:
        dataset = doc.get("dataset")

        if dataset in {"red_wine", "white_wine"}:
            if dataset not in latest_metrics:
                latest_metrics[dataset] = doc

    if not latest_metrics:
        st.warning("No model metrics were found in MongoDB.")

    else:
        available_datasets = [
            dataset
            for dataset in ["red_wine", "white_wine"]
            if dataset in latest_metrics
        ]

        selected_dataset = st.selectbox(
            "Choose a wine model to inspect",
            available_datasets,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        selected = latest_metrics[selected_dataset]
        regression = selected.get("regression", {})
        classification = selected.get("classification", {})

        st.subheader(
            f"{selected_dataset.replace('_', ' ').title()} — "
            "Evaluation Results"
        )

        # Classification metrics
        st.markdown("#### Classification Metrics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{classification.get('accuracy', 0):.2%}",
        )
        c2.metric(
            "Precision",
            f"{classification.get('precision', 0):.2%}",
        )
        c3.metric(
            "Recall",
            f"{classification.get('recall', 0):.2%}",
        )
        c4.metric(
            "F1-score",
            f"{classification.get('f1_score', 0):.2%}",
        )

        # Regression metrics
        st.markdown("#### Regression Metrics")

        r1, r2 = st.columns(2)

        r1.metric(
            "R² Score",
            f"{regression.get('r2_score', 0):.4f}",
        )
        r2.metric(
            "Mean Absolute Error (MAE)",
            f"{regression.get('mae', 0):.4f}",
        )

        # Dataset and training information
        st.markdown("#### Dataset Information")

        d1, d2, d3 = st.columns(3)

        d1.metric("Total Records", f"{selected.get('rows', 0):,}")
        d2.metric("Training Records", f"{selected.get('training_rows', 0):,}")
        d3.metric("Testing Records", f"{selected.get('testing_rows', 0):,}")

        st.caption(
            "Classification rule: quality ≥ 7 is classified as Good. "
            "R² and MAE belong to the regression model; the other four "
            "metrics belong to the classifier."
        )

        # Red versus white comparison
        if (
            "red_wine" in latest_metrics
            and "white_wine" in latest_metrics
        ):
            st.divider()
            st.subheader("🔴 Red Wine vs ⚪ White Wine")

            comparison_rows = []

            for dataset in ["red_wine", "white_wine"]:
                doc = latest_metrics[dataset]
                cls = doc.get("classification", {})
                reg = doc.get("regression", {})

                comparison_rows.append({
                    "Wine Type": dataset.replace("_", " ").title(),
                    "Records": doc.get("rows", 0),
                    "Accuracy": cls.get("accuracy"),
                    "Precision": cls.get("precision"),
                    "Recall": cls.get("recall"),
                    "F1-score": cls.get("f1_score"),
                    "R² Score": reg.get("r2_score"),
                    "MAE": reg.get("mae"),
                })

            comparison_df = pd.DataFrame(comparison_rows)

            st.dataframe(
                comparison_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-score": "{:.2%}",
                    "R² Score": "{:.4f}",
                    "MAE": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                comparison_df.set_index("Wine Type")[
                    ["Accuracy", "Precision", "Recall", "F1-score"]
                ]
            )

        # Existing charts
        st.divider()
        st.subheader("📊 Performance Visualizations")

        chart1, chart2 = st.columns(2)

        with chart1:
            regression_chart = RESULTS_DIR / "regression_metrics.png"

            if regression_chart.exists():
                st.image(
                    str(regression_chart),
                    caption="Regression Performance: R² Score and MAE",
                    use_container_width=True,
                )
            else:
                st.info("Regression chart not found in the results folder.")

        with chart2:
            classification_chart = RESULTS_DIR / "classification_metrics.png"

            if classification_chart.exists():
                st.image(
                    str(classification_chart),
                    caption="Classification Performance",
                    use_container_width=True,
                )
            else:
                st.info("Classification chart not found in the results folder.")

        confusion_chart = RESULTS_DIR / "confusion_matrices.png"

        if confusion_chart.exists():
            st.subheader("Confusion Matrices")
            st.image(
                str(confusion_chart),
                caption="Classification Confusion Matrices",
                use_container_width=True,
            )

except PyMongoError as error:
    st.error(f"Unable to retrieve model metrics from MongoDB: {error}")

st.caption(
    "Academic project: Wine Quality Prediction using Machine Learning "
    "and MongoDB."
)

