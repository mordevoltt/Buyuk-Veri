from pyspark.sql import SparkSession
import os

# Spark Session
spark = SparkSession.builder \
    .appName("Verification") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

print("\n" + "="*50)
print("SILVER TABLO (TEMİZLENMİŞ VERİ) KONTROLÜ")
print("="*50)

try:
    # Silver tabloyu oku
    df_silver = spark.read.format("delta").load("/app/data/delta/silver_table")
    
    # Toplam satır sayısı
    count = df_silver.count()
    print(f"Toplam Temizlenmiş Satır Sayısı: {count}")
    
    # Şemayı göster (Temizlik sonrası alanlar)
    print("\nVeri Yapısı (Schema):")
    df_silver.printSchema()
    
    # İlk 5 satırı göster
    print("\nİlk 5 Örnek Satır:")
    df_silver.select("timestamp", "kullanici_id", "olay_tipi", "ilgili_id").show(5, truncate=False)
    
    # Null kontrolü ispatı
    null_count = df_silver.filter("kullanici_id IS NULL OR kullanici_id = 'Anonymous'").count()
    print(f"\nTablodaki Null veya Anonymous ID Sayısı: {null_count} (0 olması temizliğin çalıştığını kanıtlar)")

except Exception as e:
    print(f"Hata oluştu: {e}")

print("="*50 + "\n")
