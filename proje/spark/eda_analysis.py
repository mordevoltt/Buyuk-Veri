from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, date_format, count, desc, when, sum as spark_sum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Spark Session ---
spark = SparkSession.builder \
    .appName("Step04_EDA_Analysis") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_PATH = os.path.join(BASE_DIR, "data", "delta", "silver_table")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
df = spark.read.format("delta").load(SILVER_PATH)

# --- 2. Basic Statistics ---
row_count = df.count()
unique_users = df.select("Customer ID").distinct().count()
unique_items = df.select("StockCode").distinct().count()

print(f"Total Rows: {row_count}")
print(f"Unique Customers: {unique_users}")
print(f"Unique Items: {unique_items}")

# General stats plot
plt.figure(figsize=(10, 4))
plt.axis('off')
summary_text = (
    f"BASIC STATISTICS SUMMARY\n"
    f"{'='*30}\n"
    f"Total Transactions  : {row_count:,}\n"
    f"Unique Customers    : {unique_users:,}\n"
    f"Unique Products     : {unique_items:,}\n"
    f"Avg Qty per Trans   : {df.selectExpr('avg(Quantity)').collect()[0][0]:.2f}\n"
    f"{'='*30}"
)
plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, family='monospace',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
plt.savefig(os.path.join(OUTPUT_DIR, "00_general_stats.png"))

# --- 3. Missing Values Analysis ---
null_counts = []
for c in df.columns:
    null_val = df.filter(col(c).isNull()).count()
    null_counts.append({'Column': c, 'Null_Count': null_val})
null_df = pd.DataFrame(null_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x='Column', y='Null_Count', data=null_df, palette='magma')
plt.title('Missing Values Analysis', fontsize=15)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_missing_values.png"))

# --- 4. Time Series Analysis ---
# Hourly Trend
hourly_data = df.withColumn("hour", hour(col("timestamp"))).groupBy("hour").count().orderBy("hour").toPandas()
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='count', data=hourly_data, marker='o', color='blue', linewidth=2.5)
plt.fill_between(hourly_data['hour'], hourly_data['count'], alpha=0.2)
plt.title('Hourly Transaction Trend', fontsize=15)
plt.savefig(os.path.join(OUTPUT_DIR, "02_hourly_trend.png"))

# Daily Trend
daily_data = df.withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd")).groupBy("date").count().orderBy("date").toPandas()
plt.figure(figsize=(15, 6))
sns.barplot(x='date', y='count', data=daily_data, color='teal')
plt.title('Daily Transaction Distribution', fontsize=15)
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, "03_daily_trend.png"))

# --- 5. Categorical & Numerical Distribution ---
# Event Types (Categorical)
event_dist = df.groupBy("Country").count().orderBy(desc("count")).limit(10).toPandas()
plt.figure(figsize=(10, 10))
plt.pie(event_dist['count'], labels=event_dist['Country'], autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title('Top 10 Countries by Transaction Count', fontsize=15)
plt.savefig(os.path.join(OUTPUT_DIR, "04_event_pie_chart.png"))

# Numerical Distribution (Quantity)
qty_dist = df.select("Quantity").limit(10000).toPandas()
plt.figure(figsize=(12, 6))
sns.histplot(qty_dist['Quantity'], bins=50, kde=True, color='green')
plt.title('Quantity Distribution Analysis', fontsize=15)
plt.xlim(0, 100) # Capped for better visibility
plt.savefig(os.path.join(OUTPUT_DIR, "05_user_activity_dist.png"))

print("EDA Analysis completed. Plots saved to:", OUTPUT_DIR)
spark.stop()
