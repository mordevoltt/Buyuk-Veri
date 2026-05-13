import os

# Base Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DELTA_PATH = os.path.join(DATA_DIR, "delta")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "checkpoints")

# ML Paths
ML_MODELS_DIR = os.path.join(PROJECT_ROOT, "ml_models")
ML_PLOTS_DIR = os.path.join(ML_MODELS_DIR, "plots")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

# Tracking
MLFLOW_EXPERIMENT_NAME = "Online_Retail_Revenue_Prediction"

# Ensure directories exist
for d in [DATA_DIR, DELTA_PATH, CHECKPOINT_PATH, ML_MODELS_DIR, ML_PLOTS_DIR, MLRUNS_DIR]:
    os.makedirs(d, exist_ok=True)
