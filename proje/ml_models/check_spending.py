from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, desc, count, mean, stddev, min, max, when, coalesce
import os

spark = SparkSession.builder \
    .appName("CheckSpending") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Veriyi oku
df = spark.read.format("delta").load("/app/data/delta/silver_table")

# Raporu oluştur
output_path = "/app/ml_models/customer_spending_report.txt"

# Kullanıcı Bazlı Harcama Analizi (Hybrid ID Mantığı)
customer_spending = df.withColumn("quantity", col("raw_data.Quantity").cast("double")) \
                      .withColumn("price", col("raw_data.Price").cast("double")) \
                      .withColumn("item_total", col("quantity") * col("price")) \
                      .withColumn("hybrid_id", when((col("kullanici_id").isNull()) | (col("kullanici_id") == "nan"), 
                                                    col("raw_data.Invoice")).otherwise(col("kullanici_id"))) \
                      .groupBy("hybrid_id").agg(
                          sum("item_total").alias("total_spending"),
                          count("*").alias("transaction_count")
                      )

# En çok harcayanları al
top_20 = customer_spending.orderBy(col("total_spending").desc()).limit(20).toPandas()

# İstatistikleri al
stats = customer_spending.select("total_spending").describe().toPandas()

with open(output_path, "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write(" 📊 MÜŞTERİ HARCAMA ANALİZ RAPORU \n")
    f.write("="*60 + "\n\n")
    
    f.write("--- GENEL İSTATİSTİKLER ---\n")
    f.write(stats.to_string(index=False) + "\n\n")
    
    f.write("--- EN ÇOK HARCAMA YAPAN İLK 20 MÜŞTERİ ---\n")
    f.write(top_20.to_string(index=False) + "\n\n")
    
    f.write("="*60 + "\n")
    f.write(f"Rapor oluşturulma zamanı: {os.popen('date').read()}")

print(f"\n✅ Rapor başarıyla oluşturuldu ve kaydedildi: {output_path}")

# Terminale de özeti basalım
customer_spending.select("total_spending").describe().show()

spark.stop()
