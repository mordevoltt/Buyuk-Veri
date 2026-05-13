import os

# ─── Temel Dizin Yolları ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Veri yolu (Workspace'e göre ayarlandı)
SILVER_DATA_PATH = os.path.join(PROJECT_DIR, "data", "delta")

# MLflow deney çıktıları
MLFLOW_TRACKING_URI = "file:///" + os.path.join(PROJECT_DIR, "mlruns").replace("\\", "/")
MLFLOW_EXPERIMENT_NAME = "Online-Retail-Dashboard"

# Dizinler
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "ml_models")
PLOTS_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "plots")

TARGET_COLUMN = "Monetary"
FEATURE_COLUMNS = ["Frequency", "TotalItemsPurchased", "Recency", "AverageOrderValue"]

MODEL_PARAMS = {
    "LinearRegression": {},
    "DecisionTreeRegressor": {},
    "RandomForestRegressor": {},
    "GBTRegressor": {},
    "GeneralizedLinearRegression": {},
}
