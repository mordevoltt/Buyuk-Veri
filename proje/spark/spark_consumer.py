import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "online_retail"

def process_batch(df, epoch_id):
    logger.info(f"--- Batch {epoch_id} işleniyor. Kayıt sayısı: {df.count()} ---")
    if df.count() == 0: return
    
    # 1. Delta Lake Bronze Tablosu (Ham temizlenmiş veri)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bronze_path = os.path.join(base_dir, "../data/bronze")
    df.write.format("delta").mode("append").save(bronze_path)
    logger.info(f"Batch {epoch_id} başarıyla Delta Lake'e (Bronze) yazıldı.")

def main():
    logger.info("Spark Session başlatılıyor...")
    
    spark = SparkSession.builder \
        .appName("OnlineRetailStreaming") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # Kafka'dan okunacak JSON şemasının tanımlanması
    schema = StructType([
        StructField("Invoice", StringType(), True), 
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True), 
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True), 
        StructField("Price", DoubleType(), True),
        StructField("Customer ID", DoubleType(), True), 
        StructField("Country", StringType(), True),
        StructField("event_timestamp", StringType(), True)
    ])

    # Kafka'dan veri okuma
    kafka_df = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .load()
        
    # JSON parse ve sütun bazlı ayırma
    json_df = kafka_df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Veri Temizleme (Null, Duplike ve Format hataları)
    cleaned_df = json_df.withColumnRenamed("Customer ID", "Customer_ID") \
                        .filter(col("Customer_ID").isNotNull()) \
                        .filter(~col("Invoice").startswith("C")) \
                        .dropDuplicates() \
                        .withColumn("TotalAmount", col("Quantity") * col("Price"))
                        
    # Micro-batch trigger
    query = cleaned_df.writeStream.foreachBatch(process_batch).outputMode("update").start()
    query.awaitTermination()

if __name__ == "__main__":
    main()
