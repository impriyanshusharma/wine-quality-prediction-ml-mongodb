# Wine Quality Prediction Using Machine Learning and MongoDB

A machine learning application that predicts wine quality using physicochemical properties. The project combines regression and binary classification models with MongoDB storage and an interactive Streamlit dashboard.

## Project Overview

This project uses the physicochemical characteristics of red and white wine to estimate quality scores and classify wines into two categories. It demonstrates a complete workflow involving data preparation, model training, evaluation, database integration, and interactive prediction.

Check the project :- https://wine-quality-prediction-ml-app-nosmxmrf33imdz9mesnupy.streamlit.app/

### Key Features

- **Machine Learning:** Random Forest regression and classification models for red and white wine.
- **Model Evaluation:** R², MAE, accuracy, precision, recall, F1-score, and confusion matrices.
- **Database Integration:** MongoDB stores datasets, evaluation metrics, and prediction history.
- **Interactive Dashboard:** Streamlit interface for predictions, visualizations, model comparisons, and prediction history.
- **Model Persistence:** Trained models are saved using Joblib and can be loaded without retraining.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Database:** MongoDB, PyMongo
- **Dashboard:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Model Storage:** Joblib

## Dataset

The project uses the Wine Quality dataset from the UCI Machine Learning Repository.

| Dataset | Number of samples |
|---|---:|
| Red wine | 1,599 |
| White wine | 4,898 |
| **Total** | **6,497** |

The dataset contains 11 physicochemical input features, including acidity, residual sugar, chlorides, density, pH, sulphates, and alcohol. Wine quality is the target variable.

**Source:** [UCI Wine Quality Dataset](https://doi.org/10.24432/C56S3T)

Dataset citation:

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository.

## Machine Learning Approach

Two types of models are trained separately for red and white wine.

### 1. Regression

Random Forest Regressor estimates a numerical wine quality score.

Evaluation metrics:
- R² Score
- Mean Absolute Error (MAE)

### 2. Binary Classification

Random Forest Classifier predicts whether a wine belongs to the good-quality category.

The classification rule used in this project is:
- **Good:** Quality score ≥ 7
- **Not Good:** Quality score < 7

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Model Performance

The following results were obtained using an 80/20 train-test split and `random_state=42`.

### Regression Results

| Dataset | R² Score | MAE |
|---|---:|---:|
| Red wine | 0.5319 | 0.4251 |
| White wine | 0.5541 | 0.4172 |

### Classification Results

| Dataset | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Red wine | 90.31% | 65.38% | 72.34% | 68.69% |
| White wine | 87.65% | 72.27% | 75.77% | 73.98% |

Classification metrics are rounded to two decimal places. Performance may vary if the data, random seed, or training configuration changes.

## Project Structure

```text
WineQualityML/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
├── src/
│   ├── load_data.py
│   ├── train_models.py
│   ├── visualize_results.py
│   └── predict_wine.py
├── models/
│   ├── red_wine_regressor.joblib
│   ├── red_wine_classifier.joblib
│   ├── white_wine_regressor.joblib
│   └── white_wine_classifier.joblib
└── results/
    ├── regression_metrics.png
    ├── classification_metrics.png
    ├── confusion_matrices.png
    ├── regression_metrics.csv
    └── classification_metrics.csv
```

The structure above represents the intended layout. Keep the filenames consistent with the files actually present in your repository.

## Installation and Setup

### Prerequisites

- Python 3.11 or another Python version compatible with the project's tested dependencies
- MongoDB Community Server
- Git

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd WineQualityML
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start MongoDB

Install and start MongoDB Community Server. Ensure the local MongoDB service is running and accessible at:

```text
mongodb://127.0.0.1:27017/
```

The current application is configured for a local MongoDB instance. If your configuration differs, update the connection settings in the relevant Python files.

### 5. Load the datasets into MongoDB

```powershell
python src/load_data.py
```

This step should populate the `red_wine` and `white_wine` collections in the `wine_quality_db` database.

Avoid repeatedly running the loading script unless it safely handles existing records.

### 6. Train the machine learning models

```powershell
python src/train_models.py
```

This step trains the regression and classification models, saves the trained model files, and stores evaluation metrics in MongoDB.

### 7. Generate evaluation visualizations

```powershell
python src/visualize_results.py
```

The generated charts and metric CSV files are saved in the `results/` directory.

### 8. Launch the Streamlit dashboard

```powershell
streamlit run app.py
```

Open the local address displayed in your terminal, typically:

```text
http://localhost:8501
```

Keep MongoDB running while using features that depend on database access.

## Dashboard

Add screenshots of your actual running application here.

Suggested screenshots:
- Main prediction interface
- Model performance and comparison
- Prediction history
- MongoDB collections in MongoDB Compass

Store the images in a folder such as `docs/images/`, then embed them in this README. For example:

```markdown
![Wine Quality Dashboard](docs/images/dashboard.png)
```

## Limitations

- Predictions are based on the dataset and trained models; they are not a definitive assessment of wine quality.
- The classification threshold of 7 converts a numerical quality score into two categories and loses some information.
- R², MAE, and classification scores describe performance on the selected test split and do not guarantee the same performance on unseen data.
- The application requires a compatible Python environment and a running MongoDB instance for its database-dependent functionality.

## Future Improvements

- Hyperparameter tuning and cross-validation
- Feature importance analysis and error analysis
- Automated tests and input validation
- FastAPI integration for serving predictions
- Docker packaging and cloud deployment
- Model monitoring and improved experiment tracking

## License and Attribution

The Wine Quality dataset is provided by the UCI Machine Learning Repository under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Give appropriate credit to the dataset authors when redistributing or adapting the data.

See the [UCI dataset page](https://doi.org/10.24432/C56S3T) for source and attribution details.

If a separate `LICENSE` file is included for this project's source code, its terms apply to your code; the dataset remains subject to its own license.

## Author

**Priyanshu Sharma**

B.Tech — Data Science
