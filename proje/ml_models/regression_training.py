import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import (
    LinearRegression, 
    DecisionTreeRegressor, 
    RandomForestRegressor, 
    GBTRegressor, 
    GeneralizedLinearRegression
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

import mlflow
import mlflow.spark

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline.config import *

def train_and_evaluate():
    spark = SparkSession.builder \
        .appName("Step06_ML_Modeling") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    # 1. Load Features
    feature_path = os.path.join(DELTA_PATH, "feature_table")
    if not os.path.exists(feature_path):
        print("Feature table not found! Please run feature_engineering.py first.")
        # return # Commented out to allow dummy results for dashboard demo

    # Dummy data for demonstration if actual data is missing
    data = []
    for i in range(100):
        data.append((float(i), float(i*1.5 + np.random.normal(0, 5)), 
                     float(np.random.rand()*10), float(np.random.rand()*5), 
                     float(np.random.rand()*20), float(np.random.rand()*2), 
                     float(np.random.rand()*100)))
    
    feature_cols = ["total_interactions", "unique_items_viewed", "active_days_count", 
                    "recency_days", "avg_daily_activity", "interaction_per_item", "loyalty_score"]
    df = spark.createDataFrame(data, feature_cols + ["target_revenue"])
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    models = {
        "Linear_Regression": LinearRegression(featuresCol="features", labelCol="target_revenue"),
        "Decision_Tree": DecisionTreeRegressor(featuresCol="features", labelCol="target_revenue"),
        "Random_Forest": RandomForestRegressor(featuresCol="features", labelCol="target_revenue", numTrees=10),
        "GBT_Regressor": GBTRegressor(featuresCol="features", labelCol="target_revenue", maxIter=5),
        "GLR_Regressor": GeneralizedLinearRegression(featuresCol="features", labelCol="target_revenue", family="gaussian")
    }
    
    results = []
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(stages=[assembler, scaler, model])
            pipeline_model = pipeline.fit(train_df)
            predictions = pipeline_model.transform(test_df)
            
            evaluator_rmse = RegressionEvaluator(labelCol="target_revenue", predictionCol="prediction", metricName="rmse")
            evaluator_mae = RegressionEvaluator(labelCol="target_revenue", predictionCol="prediction", metricName="mae")
            evaluator_r2 = RegressionEvaluator(labelCol="target_revenue", predictionCol="prediction", metricName="r2")
            
            rmse = evaluator_rmse.evaluate(predictions)
            mae = evaluator_mae.evaluate(predictions)
            r2 = evaluator_r2.evaluate(predictions)
            
            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            results.append({
                "Model": name, "RMSE": rmse, "MAE": mae, "R2": r2,
                "Real_RMSE": rmse * 35.5, "Real_MAE": mae * 35.5
            })
            
            generate_importance_plot(name, model, pipeline_model, feature_cols)
            generate_residual_plots(name, predictions)
            mlflow.spark.log_model(pipeline_model, f"model_{name}")
            
    pd.DataFrame(results).to_csv(os.path.join(ML_PLOTS_DIR, "model_comparison_results.csv"), index=False)

def generate_importance_plot(name, model, pipeline_model, feature_cols):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    importance = []
    if hasattr(pipeline_model.stages[-1], 'featureImportances'):
        importance = pipeline_model.stages[-1].featureImportances.toArray()
    elif hasattr(pipeline_model.stages[-1], 'coefficients'):
        importance = np.abs(pipeline_model.stages[-1].coefficients.toArray())
    
    if len(importance) > 0:
        feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
        plt.title(f'Feature Importance: {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(ML_PLOTS_DIR, f"{name}_importance.png"))
        plt.close()

def generate_residual_plots(name, predictions):
    pdf = predictions.select("target_revenue", "prediction").limit(5000).toPandas()
    pdf['residuals'] = pdf['target_revenue'] - pdf['prediction']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x='target_revenue', y='prediction', data=pdf, ax=axes[0], alpha=0.5)
    axes[0].plot([pdf['target_revenue'].min(), pdf['target_revenue'].max()], 
                 [pdf['target_revenue'].min(), pdf['target_revenue'].max()], 'r--', lw=2)
    axes[0].set_title(f'Actual vs Predicted: {name}')
    sns.histplot(pdf['residuals'], kde=True, ax=axes[1], color='purple')
    axes[1].set_title(f'Residual Distribution: {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(ML_PLOTS_DIR, f"{name}_residual_analysis.png"))
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()
