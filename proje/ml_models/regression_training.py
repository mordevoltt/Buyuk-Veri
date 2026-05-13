import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, when, expm1
from pyspark.ml.feature import VectorAssembler, RobustScaler, PolynomialExpansion
from pyspark.ml.regression import (
    LinearRegression, DecisionTreeRegressor, RandomForestRegressor,
    GBTRegressor, GeneralizedLinearRegression
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline, PipelineModel
import mlflow
import mlflow.spark
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import logging

# Optuna loglarını biraz sessize alalım ki kalabalık etmesin
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. Spark Session Yapılandırması ---
spark = SparkSession.builder \
    .appName("UltimatePremiumRegression") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# MLflow Ayarları
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Regression_Final_Production_V1")
mlflow.spark.autolog(disable=True)

# Debug Log Dosyası
debug_log_path = "/app/ml_models/mlflow_debug.txt"
debug_log = open(debug_log_path, "w", encoding="utf-8")

def log(msg):
    """Hem konsola hem txt dosyasına yazar."""
    print(msg)
    debug_log.write(msg + "\n")
    debug_log.flush()

# --- 2. Veri Okuma ve Premium Özellik Mühendisliği ---
log("🚀 Veri okunuyor ve 'Premium' özellikler türetiliyor...")
df = spark.read.format("delta").load("/app/data/delta/feature_table")

# Null Doldurma ve Temizlik
df = df.withColumn("recency_days", when(col("recency_days").isNull(), 365).otherwise(col("recency_days")))
df = df.filter(col("total_spending") >= 0)
df = df.fillna(0)

# Ham Kolonlar
base_cols = ["total_interactions", "unique_items_viewed", "active_days_count", "recency_days"]

# PREMIUM HAMLE 1: Oransal ve Değer Odaklı Özellikler
df = df.withColumn("avg_daily_activity", col("total_interactions") / (col("active_days_count") + 1))
df = df.withColumn("interaction_per_item", col("total_interactions") / (col("unique_items_viewed") + 1))
df = df.withColumn("loyalty_score", col("active_days_count") / (col("recency_days") + 1))
# Müşteri Değer Skoru (RFM Mantığı)
df = df.withColumn("customer_value_score", (col("total_interactions") * col("active_days_count")) / (col("recency_days") + 1))

tree_cols = base_cols + ["avg_daily_activity", "interaction_per_item", "loyalty_score", "customer_value_score"]

# Hedef Değişken Log Dönüşümü
df = df.withColumn("target", log1p(col("total_spending")))
target_col = "target"

# --- 3. Veri Bölme ve Aykırı Değer Tıraşlama ---
log("🛡️ Veri bölünüyor ve Aykırı Değerler Train seti üzerinden tıraşlanıyor...")

# ÖNCE Bölme (Sızıntıyı önlemek için)
train_raw, test_raw = df.randomSplit([0.85, 0.15], seed=42)

# SONRA Tıraşlama (Sadece orijinal kolonları güncelliyoruz)
for c in tree_cols:
    quantiles = train_raw.approxQuantile(c, [0.01, 0.99], 0.0)
    
    if len(quantiles) < 2:
        log(f"⚠️  {c} kolonu için yeterli veri yok, tıraşlama atlanıyor.")
        continue
        
    q_lower, q_upper = quantiles[0], quantiles[1]
    train_raw = train_raw.withColumn(c, when(col(c) > q_upper, q_upper).when(col(c) < q_lower, q_lower).otherwise(col(c)))
    test_raw = test_raw.withColumn(c, when(col(c) > q_upper, q_upper).when(col(c) < q_lower, q_lower).otherwise(col(c)))

# --- 4. Vektörleştirme ve Polinom Açılımı (TIRAŞLANMIŞ VERİ İLE) ---
log("🛡️ Tıraşlanmış veriler üzerinden Vektörleştirme ve Polinom açılımı yapılıyor...")

assembler_lin = VectorAssembler(inputCols=base_cols, outputCol="raw_features_lin")
polyExpansion = PolynomialExpansion(degree=2, inputCol="raw_features_lin", outputCol="poly_features_lin")
assembler_tree = VectorAssembler(inputCols=tree_cols, outputCol="raw_features_tree")

def prepare_features(data_df):
    d1 = assembler_lin.transform(data_df)
    d2 = polyExpansion.transform(d1)
    d3 = assembler_tree.transform(d2)
    return d3

train_df = prepare_features(train_raw)
test_df = prepare_features(test_raw)

# RobustScaler Tanımı (Pipeline içinde kullanılacak)
scaler = RobustScaler(inputCol="poly_features_lin", outputCol="scaled_poly_features")

# --- 5. Modeller ve Genişletilmiş Parametreler ---
models = [
    ("Linear_Regression", LinearRegression(labelCol=target_col, featuresCol="scaled_poly_features", solver="l-bfgs"), 
     ParamGridBuilder().addGrid(LinearRegression.regParam, [0.1, 1.0]).addGrid(LinearRegression.elasticNetParam, [0.5, 1.0]).build()),
    
    ("Decision_Tree", DecisionTreeRegressor(labelCol=target_col, featuresCol="raw_features_tree"), 
     ParamGridBuilder().addGrid(DecisionTreeRegressor.maxDepth, [5, 10]).build()),
    
    ("Random_Forest", RandomForestRegressor(labelCol=target_col, featuresCol="raw_features_tree"), 
     ParamGridBuilder().addGrid(RandomForestRegressor.numTrees, [50, 100]).addGrid(RandomForestRegressor.maxDepth, [5, 8]).build()),
    
    ("GBT_Regressor", GBTRegressor(labelCol=target_col, featuresCol="raw_features_tree"), 
     ParamGridBuilder().addGrid(GBTRegressor.maxIter, [50, 100]).addGrid(GBTRegressor.maxDepth, [5, 7]).build()),
    
    ("GLR", GeneralizedLinearRegression(labelCol=target_col, featuresCol="scaled_poly_features", family="gaussian"), 
     ParamGridBuilder().addGrid(GeneralizedLinearRegression.regParam, [0.1, 0.5]).build())
]

eval_rmse = RegressionEvaluator(labelCol=target_col, metricName="rmse")
eval_mae = RegressionEvaluator(labelCol=target_col, metricName="mae")
eval_r2 = RegressionEvaluator(labelCol=target_col, metricName="r2")

plot_dir = "/app/ml_models/plots"
os.makedirs(plot_dir, exist_ok=True)

# Sonuçları toplamak için liste
all_results = []

log("\n" + "="*60)
log(" 🎓 ULTIMATE PREMIUM TRAINING STARTING (LEAK-FREE & BUG-FIXED) ")
log("="*60)

for name, model_obj, paramGrid in models:
    with mlflow.start_run(run_name=name):
        log(f"\n>>> {name} Eğitimi Başlıyor...")
        
        # ========== MODEL EĞİTİMİ ==========
        best_params_dict = {}
        
        if name in ["Decision_Tree", "Random_Forest", "GBT_Regressor"]:
            log(f"   🔍 Optuna + 3-Fold CV ile derin optimizasyon yapılıyor...")
            def objective(trial):
                try:
                    if name == "Decision_Tree":
                        m_obj = DecisionTreeRegressor(labelCol=target_col, featuresCol="raw_features_tree", maxDepth=trial.suggest_int("maxDepth", 2, 15))
                    elif name == "Random_Forest":
                        m_obj = RandomForestRegressor(labelCol=target_col, featuresCol="raw_features_tree", numTrees=trial.suggest_int("numTrees", 100, 300), maxDepth=trial.suggest_int("maxDepth", 4, 12))
                    elif name == "GBT_Regressor":
                        m_obj = GBTRegressor(labelCol=target_col, featuresCol="raw_features_tree", maxIter=trial.suggest_int("maxIter", 20, 100), maxDepth=trial.suggest_int("maxDepth", 3, 10))
                    
                    cv = CrossValidator(estimator=m_obj, estimatorParamMaps=ParamGridBuilder().build(), evaluator=eval_rmse, numFolds=3)
                    cv_model = cv.fit(train_df)
                    return cv_model.avgMetrics[0]
                except Exception as e:
                    return 9999.0

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)
            best_params_dict = study.best_params
            
            if name == "Decision_Tree":
                best_model_obj = DecisionTreeRegressor(labelCol=target_col, featuresCol="raw_features_tree", **best_params_dict)
            elif name == "Random_Forest":
                best_model_obj = RandomForestRegressor(labelCol=target_col, featuresCol="raw_features_tree", **best_params_dict)
            elif name == "GBT_Regressor":
                best_model_obj = GBTRegressor(labelCol=target_col, featuresCol="raw_features_tree", **best_params_dict)
            
            best_model = best_model_obj.fit(train_df)
        else:
            log(f"   📉 Pipeline + CrossValidator uygulanıyor...")
            pipeline = Pipeline(stages=[scaler, model_obj])
            cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=eval_rmse, numFolds=3)
            cv_model = cv.fit(train_df)
            best_model = cv_model.bestModel
            # Pipeline modellerinden parametreleri çıkar
            actual_m = best_model.stages[-1]
            if name == "Linear_Regression":
                best_params_dict = {"regParam": actual_m.getOrDefault("regParam"),
                                    "elasticNetParam": actual_m.getOrDefault("elasticNetParam"),
                                    "solver": "l-bfgs"}
            elif name == "GLR":
                best_params_dict = {"regParam": actual_m.getOrDefault("regParam"),
                                    "family": "gaussian"}
        
        # ========== PARAMETRELERİ MLFLOW'A LOGLA ==========
        for param_name, param_val in best_params_dict.items():
            mlflow.log_param(param_name, param_val)
        mlflow.log_param("model_type", name)
        mlflow.log_param("cross_validation_folds", 3)
        mlflow.log_param("train_ratio", 0.85)
        mlflow.log_param("test_ratio", 0.15)
        
        # ========== TEST VE EVALUATION ==========
        test_preds = best_model.transform(test_df)
        rmse = eval_rmse.evaluate(test_preds)
        mae = eval_mae.evaluate(test_preds)
        r2 = eval_r2.evaluate(test_preds)
        
        # Gerçek Dünya Metrikleri
        real_preds = test_preds.withColumn("real_target", expm1(col(target_col))) \
                               .withColumn("real_prediction", expm1(col("prediction")))
        real_rmse = RegressionEvaluator(labelCol="real_target", predictionCol="real_prediction", metricName="rmse").evaluate(real_preds)
        real_mae = RegressionEvaluator(labelCol="real_target", predictionCol="real_prediction", metricName="mae").evaluate(real_preds)
        
        log(f"   [OK] {name} -> Log RMSE: {rmse:.4f} | R2: {r2:.4f}")
        log(f"   [REAL] {name} -> Gerçek RMSE (Para): {real_rmse:.2f} | Gerçek MAE (Para): {real_mae:.2f}")

        # ========== TÜM METRİKLERİ MLFLOW'A LOGLA ==========
        mlflow.log_metrics({
            "rmse": rmse, "mae": mae, "r2": r2,
            "real_rmse": real_rmse, "real_mae": real_mae
        })
        
        # Modeli kaydet
        try:
            mlflow.spark.log_model(best_model, "model")
            log(f"   ✅ Model MLflow'a kaydedildi.")
        except Exception as e:
            log(f"   ⚠️ Model kaydetme hatası (devam ediliyor): {e}")
            try:
                model_path = f"/app/ml_models/saved_models/{name}"
                best_model.save(model_path)
                mlflow.log_param("model_local_path", model_path)
                log(f"   ✅ Model lokale kaydedildi: {model_path}")
            except Exception as e2:
                log(f"   ⚠️ Lokal model kaydetme de başarısız: {e2}")

        # Sonuçları karşılaştırma tablosuna ekle
        all_results.append({
            "Model": name, "RMSE": rmse, "MAE": mae, "R2": r2,
            "Real_RMSE": real_rmse, "Real_MAE": real_mae
        })

        # ========== RESIDUAL (ARTIK) ANALİZİ ==========
        log(f"   📊 {name} için Residual analizi yapılıyor...")
        residual_pdf = test_preds.select(
            col(target_col).alias("actual"),
            col("prediction")
        ).toPandas()
        residual_pdf["residual"] = residual_pdf["actual"] - residual_pdf["prediction"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Residual Analizi - {name}", fontsize=16, fontweight="bold")
        
        # 1. Residual vs Predicted
        axes[0, 0].scatter(residual_pdf["prediction"], residual_pdf["residual"],
                           alpha=0.4, s=10, c="steelblue")
        axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[0, 0].set_xlabel("Tahmin (Predicted)")
        axes[0, 0].set_ylabel("Artık (Residual)")
        axes[0, 0].set_title("Residual vs Predicted")
        
        # 2. Residual Histogram
        axes[0, 1].hist(residual_pdf["residual"], bins=50, color="steelblue",
                        edgecolor="white", alpha=0.8)
        axes[0, 1].set_xlabel("Artık (Residual)")
        axes[0, 1].set_ylabel("Frekans")
        axes[0, 1].set_title("Residual Dağılımı")
        axes[0, 1].axvline(x=0, color="red", linestyle="--", linewidth=1)
        
        # 3. Actual vs Predicted
        axes[1, 0].scatter(residual_pdf["actual"], residual_pdf["prediction"],
                           alpha=0.4, s=10, c="darkgreen")
        min_val = min(residual_pdf["actual"].min(), residual_pdf["prediction"].min())
        max_val = max(residual_pdf["actual"].max(), residual_pdf["prediction"].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        axes[1, 0].set_xlabel("Gerçek Değer (Actual)")
        axes[1, 0].set_ylabel("Tahmin (Predicted)")
        axes[1, 0].set_title("Actual vs Predicted")
        
        # 4. QQ Plot (Residual Normallik)
        sorted_residuals = np.sort(residual_pdf["residual"].values)
        n = len(sorted_residuals)
        theoretical_q = np.array([np.percentile(np.random.standard_normal(10000), 
                                  (i + 0.5) / n * 100) for i in range(n)])
        axes[1, 1].scatter(theoretical_q, sorted_residuals, alpha=0.4, s=10, c="purple")
        axes[1, 1].plot([theoretical_q.min(), theoretical_q.max()],
                        [theoretical_q.min(), theoretical_q.max()], "r--", linewidth=1)
        axes[1, 1].set_xlabel("Teorik Quantile")
        axes[1, 1].set_ylabel("Gözlenen Quantile")
        axes[1, 1].set_title("Q-Q Plot (Normallik Testi)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        residual_path = f"{plot_dir}/{name}_residual_analysis.png"
        plt.savefig(residual_path, dpi=150)
        plt.close()
        try:
            mlflow.log_artifact(residual_path)
            log(f"   ✅ Residual analizi MLflow'a kaydedildi: {residual_path}")
        except Exception as e:
            log(f"   ⚠️ Residual artifact hatası (dosya lokalde mevcut): {e}")
            log(f"   📁 Lokal dosya: {residual_path}")

        # ========== FEATURE IMPORTANCE ==========
        actual_model = best_model.stages[-1] if isinstance(best_model, PipelineModel) else best_model
        importances = None
        current_cols = None
        
        if hasattr(actual_model, "featureImportances"):
            importances = actual_model.featureImportances.toArray()
            current_cols = tree_cols
        elif hasattr(actual_model, "coefficients"):
            importances = [abs(x) for x in actual_model.coefficients.toArray()]
            current_cols = [f"poly_feat_{i}" for i in range(len(importances))]
            
        if importances is not None and current_cols is not None:
            # En fazla top 10 veya mevcut feature sayısı kadar göster
            n_features = min(10, len(current_cols))
            feat_df = pd.DataFrame({
                'Feature': current_cols[:len(importances)],
                'Importance': importances[:len(current_cols)]
            }).sort_values(by='Importance', ascending=False).head(n_features)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette="viridis")
            plt.title(f'Top {n_features} Feature Importance - {name}', fontsize=14, fontweight="bold")
            plt.xlabel("Önem Derecesi (Importance)")
            plt.ylabel("Özellik (Feature)")
            plt.tight_layout()
            plot_path = f"{plot_dir}/{name}_importance.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            try:
                mlflow.log_artifact(plot_path)
                log(f"   ✅ Feature Importance MLflow'a kaydedildi.")
            except Exception as e:
                log(f"   ⚠️ FI artifact hatası (dosya lokalde mevcut): {e}")
            
            # Feature importance değerlerini MLflow'a param/metric olarak logla
            for idx, row in feat_df.iterrows():
                safe_name = row['Feature'].replace(" ", "_")[:50]
                mlflow.log_metric(f"fi_{safe_name}", float(row['Importance']))
            
            print(f"   ✅ Feature Importance kaydedildi: {plot_path}")
        else:
            log(f"   ⚠️ {name} modeli için Feature Importance çıkarılamadı.")

# ========== MODEL KARŞILAŞTIRMA GRAFİĞİ ==========
log("\n📊 Model Karşılaştırma Grafikleri oluşturuluyor...")
results_df = pd.DataFrame(all_results)

# Karşılaştırma tablosunu kaydet
results_csv_path = f"{plot_dir}/model_comparison_results.csv"
results_df.to_csv(results_csv_path, index=False)
log(f"   ✅ Karşılaştırma tablosu: {results_csv_path}")

# 1. Metrik Karşılaştırma Grafiği (RMSE, MAE, R²)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Model Karşılaştırma - Regresyon Metrikleri", fontsize=16, fontweight="bold")

colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

# RMSE
bars1 = axes[0].bar(results_df["Model"], results_df["RMSE"], color=colors, edgecolor="white")
axes[0].set_title("RMSE Karşılaştırması", fontsize=13)
axes[0].set_ylabel("RMSE (Log Scale)")
axes[0].tick_params(axis='x', rotation=25)
for bar, val in zip(bars1, results_df["RMSE"]):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# MAE
bars2 = axes[1].bar(results_df["Model"], results_df["MAE"], color=colors, edgecolor="white")
axes[1].set_title("MAE Karşılaştırması", fontsize=13)
axes[1].set_ylabel("MAE (Log Scale)")
axes[1].tick_params(axis='x', rotation=25)
for bar, val in zip(bars2, results_df["MAE"]):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# R²
bars3 = axes[2].bar(results_df["Model"], results_df["R2"], color=colors, edgecolor="white")
axes[2].set_title("R² Score Karşılaştırması", fontsize=13)
axes[2].set_ylabel("R² Score")
axes[2].tick_params(axis='x', rotation=25)
for bar, val in zip(bars3, results_df["R2"]):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
comparison_path = f"{plot_dir}/model_comparison_metrics.png"
plt.savefig(comparison_path, dpi=150)
plt.close()
print(f"   ✅ Metrik karşılaştırma grafiği: {comparison_path}")

# 2. Gerçek Değer Metrik Karşılaştırması
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Model Karşılaştırma - Gerçek Değer Metrikleri (₺)", fontsize=16, fontweight="bold")

bars4 = axes2[0].bar(results_df["Model"], results_df["Real_RMSE"], color=colors, edgecolor="white")
axes2[0].set_title("Gerçek RMSE (₺)", fontsize=13)
axes2[0].set_ylabel("RMSE (Para)")
axes2[0].tick_params(axis='x', rotation=25)
for bar, val in zip(bars4, results_df["Real_RMSE"]):
    axes2[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                  f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

bars5 = axes2[1].bar(results_df["Model"], results_df["Real_MAE"], color=colors, edgecolor="white")
axes2[1].set_title("Gerçek MAE (₺)", fontsize=13)
axes2[1].set_ylabel("MAE (Para)")
axes2[1].tick_params(axis='x', rotation=25)
for bar, val in zip(bars5, results_df["Real_MAE"]):
    axes2[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                  f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
real_comparison_path = f"{plot_dir}/model_comparison_real_metrics.png"
plt.savefig(real_comparison_path, dpi=150)
plt.close()
print(f"   ✅ Gerçek değer karşılaştırma grafiği: {real_comparison_path}")

# Karşılaştırma grafiklerini de MLflow'a logla (ayrı bir parent run ile)
with mlflow.start_run(run_name="Model_Comparison_Summary"):
    for artifact_path in [comparison_path, real_comparison_path, results_csv_path]:
        try:
            mlflow.log_artifact(artifact_path)
            log(f"   ✅ Artifact kaydedildi: {artifact_path}")
        except Exception as e:
            log(f"   ⚠️ Artifact hatası ({artifact_path}): {e}")
    # En iyi modeli logla
    best_idx = results_df["R2"].idxmax()
    best_model_name = results_df.loc[best_idx, "Model"]
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_r2", results_df.loc[best_idx, "R2"])
    mlflow.log_metric("best_rmse", results_df.loc[best_idx, "RMSE"])
    mlflow.log_metric("best_mae", results_df.loc[best_idx, "MAE"])
    print(f"\n🏆 En İyi Model: {best_model_name} (R²: {results_df.loc[best_idx, 'R2']:.4f})")
    log(f"🏆 En İyi Model: {best_model_name} (R²: {results_df.loc[best_idx, 'R2']:.4f})")

# Sonuç Tablosunu Ekrana Yazdır
log("\n" + "="*80)
log(" 📋 MODEL KARŞILAŞTIRMA SONUÇLARI")
log("="*80)
log(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Real RMSE':>12} {'Real MAE':>12}")
log("-"*80)
for _, row in results_df.iterrows():
    log(f"{row['Model']:<22} {row['RMSE']:>10.4f} {row['MAE']:>10.4f} {row['R2']:>10.4f} {row['Real_RMSE']:>12.2f} {row['Real_MAE']:>12.2f}")
log("="*80)

log("\n✅ TÜM MODELLER TAMAMLANDI! MLflow Dashboard'dan sonuçları inceleyebilirsiniz.")
log(f"📄 Debug log dosyası: {debug_log_path}")
debug_log.close()
spark.stop()
