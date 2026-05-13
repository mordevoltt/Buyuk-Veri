from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, datediff, max, min, to_timestamp, sum, when, expr
from pyspark.sql.window import Window
import os

"""
ADIM 5: ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)

Bu aşamada, ML modeline beslenecek 5'ten fazla anlamlı özellik (feature) üretilmektedir.
Veri sızıntısını (data leakage) önlemek için veriler %80 geçmiş, %20 gelecek şeklinde bölünmüştür.

Üretilen Özellikler ve İş Mantığı:
1. total_interactions: Kullanıcının toplam işlem sayısı. Sadakat ve aktivite seviyesini gösterir.
2. unique_items_viewed: Kullanıcının etkileşime girdiği benzersiz ürün sayısı. İlgi alanının çeşitliliğini gösterir.
3. active_days_count: Kullanıcının aktif olduğu gün sayısı. Düzenli bir kullanıcı olup olmadığını gösterir.
4. recency_days: Kullanıcının son işleminden bu yana geçen gün sayısı. Güncel bir kullanıcı olup olmadığını gösterir.
5. avg_daily_activity: Gün başına düşen ortalama işlem sayısı. Yoğunluk ölçüsüdür.
6. interaction_per_item: Ürün başına düşen ortalama işlem sayısı. Belirli ürünlere olan bağlılığı gösterir.
7. loyalty_score: Aktif günlerin, son görülme tarihine oranı. Kullanıcının tutarlılığını ölçer.
"""

# --- 1. Spark Session Yapılandırması ---
spark = SparkSession.builder \
    .appName("FeatureEngineeringStep") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# --- 2. Veriyi Oku (Silver Katmanı) ---
silver_path = "/app/data/delta/silver_table"
print(f"\n[INFO] Silver tablodan veri okunuyor: {silver_path}")

# Local ortamda çalışıyorsak yolu düzeltelim (opsiyonel, docker içinde çalışıyorsa /app/ kalabilir)
if not os.path.exists(silver_path) and os.path.exists("data/delta/silver_table"):
    silver_path = "data/delta/silver_table"

df = spark.read.format("delta").load(silver_path)

# Zaman damgasını ve Hybrid ID'yi hazırla
df = df.withColumn("ts", to_timestamp(col("timestamp"))) \
       .withColumn("hybrid_id", when((col("kullanici_id").isNull()) | (col("kullanici_id") == "nan"), 
                                     col("raw_data.Invoice")).otherwise(col("kullanici_id")))

# --- 3. Dinamik Global Zaman Bölmesi (Sızıntı Sıfırlama) ---
print("[INFO] Dinamik Zaman Tüneli hesaplanıyor...")

time_stats = df.select(
    min("ts").alias("min_ts"),
    max("ts").alias("max_ts")
).collect()[0]

min_ts = time_stats["min_ts"]
max_ts = time_stats["max_ts"]

# Toplam süreyi saniye cinsinden bulup %80'ini hesaplayalım
total_seconds = (max_ts.timestamp() - min_ts.timestamp())
cut_off_seconds = total_seconds * 0.8
global_cut_off = min_ts.fromtimestamp(min_ts.timestamp() + cut_off_seconds)

print(f"[INFO] Veri Başlangıcı: {min_ts}")
print(f"[INFO] Veri Bitişi   : {max_ts}")
print(f"[INFO] Dinamik Kesim Noktası (Cut-off): {global_cut_off}")

# Tüm veriyi bu "Bıçak Sırtı" noktasından ikiye bölelim
past_df = df.filter(col("ts") < global_cut_off)
future_df = df.filter(col("ts") >= global_cut_off)

# --- 4. Özellikleri Hesapla (Past DF üzerinden) ---
print("[INFO] Geçmiş (%80 zaman) verilerinden özellikler hesaplanıyor...")
features_df = past_df.withColumn("quantity", col("raw_data.Quantity").cast("double")) \
                     .withColumn("price", col("raw_data.Price").cast("double")) \
                     .withColumn("item_total", col("quantity") * col("price")) \
                     .groupBy("hybrid_id").agg(
    count("*").alias("total_interactions"),
    countDistinct("ilgili_id").alias("unique_items_viewed"),
    countDistinct(col("timestamp").substr(1, 10)).alias("active_days_count"),
    max("ts").alias("last_seen_ts_past")
)

# Özellik Türetme Mantığı
features_df = features_df.withColumn("recency_days", datediff(expr(f"cast('{global_cut_off}' as timestamp)"), col("last_seen_ts_past")))
features_df = features_df.withColumn("avg_daily_activity", col("total_interactions") / (col("active_days_count") + 1))
features_df = features_df.withColumn("interaction_per_item", col("total_interactions") / (col("unique_items_viewed") + 1))
features_df = features_df.withColumn("loyalty_score", col("active_days_count") / (col("recency_days") + 1))

# --- 5. Hedefi Hesapla (Future DF üzerinden) ---
print("[INFO] Gelecek (%20) veriden hedef değişken (Spending) hesaplanıyor...")
target_df = future_df.withColumn("quantity", col("raw_data.Quantity").cast("double")) \
                     .withColumn("price", col("raw_data.Price").cast("double")) \
                     .withColumn("item_total", col("quantity") * col("price")) \
                     .groupBy("hybrid_id").agg(sum("item_total").alias("total_spending"))

# --- 6. Birleştirme (Join) ---
final_df = features_df.join(target_df, on="hybrid_id", how="left").fillna(0, ["total_spending"])
final_df = final_df.withColumnRenamed("hybrid_id", "kullanici_id").drop("last_seen_ts_past")

print("\n" + "="*50)
print(" ÖZELLİK TABLOSU ÖRNEĞİ (ADIM 5) ")
print("="*50)
final_df.show(10)

# --- 7. Kaydetme (Gold Layer) ---
feature_output_path = "/app/data/delta/feature_table"

# Local ortam kontrolü
if not os.path.exists("/app") and os.path.exists("data"):
    feature_output_path = "data/delta/feature_table"

print(f"\n[INFO] Özellik tablosu Delta Lake'e kaydediliyor: {feature_output_path}")

final_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(feature_output_path)

print("\n✅ ADIM 5 TAMAMLANDI: 7 farklı özellik Delta Lake'e kaydedildi.")
spark.stop()
