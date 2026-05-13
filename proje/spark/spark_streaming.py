from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, MapType
import logging
import os

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/spark_processor.log')
    ]
)
logger = logging.getLogger("SparkProcessor")

# 1. Spark Session Yapılandırması
# Delta Lake ve Kafka paketlerini çalışma anında ekleyeceğiz (npx/spark-submit ile veya config ile)
spark = SparkSession.builder \
    .appName("OnlineRetailStreamingProcessor") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Şema Tanımlama
# Producer'dan gelen JSON yapısına göre: timestamp, kullanici_id, olay_tipi, ilgili_id, raw_data
schema = StructType([
    StructField("timestamp", StringType()),
    StructField("kullanici_id", StringType()),
    StructField("olay_tipi", StringType()),
    StructField("ilgili_id", StringType()),
    StructField("raw_data", MapType(StringType(), StringType()))
])

# 3. Kafka'dan Veri Okuma
logger.info("Kafka'dan veri okunuyor...")
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "online_retail") \
    .option("startingOffsets", "earliest") \
    .load()

# 4. BRONZE KATMANI: Ham Veri Yazımı
# Kafka'dan gelen ham bayt veriyi string'e çevirip meta verilerle birlikte saklıyoruz.
bronze_df = kafka_df.selectExpr("CAST(value AS STRING) as raw_value", "timestamp as kafka_timestamp", "offset", "partition")

bronze_query = bronze_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/app/data/checkpoints/bronze") \
    .start("/app/data/delta/bronze_table")

# 5. SILVER KATMANI: Veri Temizleme ve Parse Etme
# JSON parse ediliyor ve tipler düzenleniyor
silver_df = kafka_df.select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("processed_timestamp", current_timestamp())

# Veri Temizleme İşlemleri:
# - Null 'kullanici_id' olanları filtrele
# - Duplike kayıtları temizle (Watermark kullanarak)
silver_cleaned_df = silver_df \
    .withWatermark("processed_timestamp", "10 minutes") \
    .filter(col("kullanici_id").isNotNull() & (col("kullanici_id") != "Anonymous")) \
    .dropDuplicates(["kullanici_id", "timestamp"])

silver_query = silver_cleaned_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/app/data/checkpoints/silver") \
    .start("/app/data/delta/silver_table")

# 6. GOLD KATMANI: Analitik ve Agregasyon (Örn: Saniyede kaç olay oluyor?)
gold_df = silver_cleaned_df \
    .groupBy(window(col("processed_timestamp"), "1 minute"), "olay_tipi") \
    .count()

gold_query = gold_df.writeStream \
    .format("delta") \
    .outputMode("complete") \
    .option("checkpointLocation", "/app/data/checkpoints/gold") \
    .start("/app/data/delta/gold_table")

logger.info("Stream işlemleri başlatıldı. Bekleniyor...")
spark.streams.awaitAnyTermination()
