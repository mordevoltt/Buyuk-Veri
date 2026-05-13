from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, current_timestamp, when, window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
import os

# --- 1. Spark Session ---
spark = SparkSession.builder \
    .appName("Step03_Spark_Streaming_Ingestion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "delta")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "data", "checkpoints")

BRONZE_PATH = os.path.join(DATA_DIR, "bronze_table")
SILVER_PATH = os.path.join(DATA_DIR, "silver_table")
GOLD_PATH = os.path.join(DATA_DIR, "gold_table")

# --- 2. Schema Definition ---
# Matching Online Retail II structure
schema = StructType([
    StructField("Invoice", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Quantity", LongType(), True),
    StructField("InvoiceDate", StringType(), True),
    StructField("Price", DoubleType(), True),
    StructField("Customer ID", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("event_type", StringType(), True)
])

# --- 3. Kafka Source ---
# Note: Using 'readStream' for Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "online_retail_topic") \
    .option("startingOffsets", "earliest") \
    .load()

# --- 4. Bronze Layer (Raw Ingestion) ---
bronze_df = kafka_df.selectExpr("CAST(value AS STRING) as raw_payload", "timestamp as ingestion_time")

# Write to Bronze
bronze_query = bronze_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "bronze")) \
    .start(BRONZE_PATH)

# --- 5. Silver Layer (Parsing & Cleaning) ---
silver_df = bronze_df.withColumn("data", from_json(col("raw_payload"), schema)) \
    .select("data.*") \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .withColumn("Price", col("Price").cast("double")) \
    .withColumn("Quantity", col("Quantity").cast("long"))

# Cleaning Logic:
# - Remove records with null essential fields
# - Filter out quantity <= 0 (returns/errors)
# - Drop duplicates based on Invoice and StockCode
cleaned_silver_df = silver_df.filter(
    col("Invoice").isNotNull() & 
    col("Price").isNotNull() & 
    (col("Quantity") > 0)
).dropDuplicates(["Invoice", "StockCode", "timestamp"])

# Write to Silver
silver_query = cleaned_silver_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "silver")) \
    .start(SILVER_PATH)

# --- 6. Gold Layer (Business Aggregations) ---
# Goal: Calculate real-time revenue and transaction counts
gold_df = cleaned_silver_df \
    .withColumn("revenue", col("Quantity") * col("Price")) \
    .groupBy(
        window(col("timestamp"), "1 hour"),
        col("Country")
    ).agg({
        "revenue": "sum",
        "Invoice": "count"
    }).withColumnRenamed("sum(revenue)", "total_revenue") \
      .withColumnRenamed("count(Invoice)", "transaction_count")

# Write to Gold
gold_query = gold_df.writeStream \
    .format("delta") \
    .outputMode("complete") \
    .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "gold")) \
    .start(GOLD_PATH)

# --- 7. Monitoring ---
print("Streaming queries started...")
spark.streams.awaitAnyTermination()
