from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, date_format, count, desc, coalesce, when
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- 1. Spark Session Yapılandırması ---
spark = SparkSession.builder \
    .appName("DeltaLakeComprehensiveEDA") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.ui.showConsoleProgress", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Çıktı klasörü ayarları
output_dir = "/app/data/plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Stil Ayarları
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "🚀" * 20)
print("  KAPSAMLI EDA İŞLEMİ BAŞLATILIYOR (GÖRSEL ODAKLI)  ")
print("🚀" * 20 + "\n")

# --- 2. Veriyi Yükle ---
silver_path = "/app/data/delta/silver_table"
df = spark.read.format("delta").load(silver_path)
df = df.withColumn("ts", to_timestamp(col("timestamp"))) \
       .withColumn("final_id", when((col("kullanici_id").isNull()) | (col("kullanici_id") == "nan"), 
                                    col("raw_data.Invoice")).otherwise(col("kullanici_id")))

# --- 3. Temel İstatistikleri Hesapla ---
row_count = df.count()
unique_users = df.select("final_id").distinct().count()
unique_events = df.select("olay_tipi").distinct().count()

# YENİ: GENEL ÖZET GRAFİĞİ (00_general_stats.png)
plt.figure(figsize=(8, 4))
plt.axis('off')
summary_text = (
    f"GENEL VERİ ÖZETİ\n"
    f"{'-'*30}\n"
    f"Toplam Kayıt Sayısı      : {row_count:,}\n"
    f"Benzersiz Kullanıcı      : {unique_users:,}\n"
    f"Benzersiz Olay Tipi      : {unique_events}\n"
    f"{'-'*30}\n"
    f"Analiz Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
)
plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=18, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue', boxstyle='round,pad=1'))
plt.savefig(f"{output_dir}/00_general_stats.png")
print("[OK] Genel özet grafiği oluşturuldu.")

# --- 4. Eksik Değer Analizi ---
null_counts = []
for c in df.columns:
    c_null = df.filter(col(c).isNull()).count()
    null_counts.append({'Sütun': c, 'Eksik_Sayisi': c_null})
null_df = pd.DataFrame(null_counts)

plt.figure(figsize=(10, 5))
if null_df['Eksik_Sayisi'].sum() > 0:
    sns.barplot(x='Sütun', y='Eksik_Sayisi', data=null_df, palette='Reds_r')
else:
    plt.text(0.5, 0.5, 'Tebrikler!\nHiç Eksik Değer Bulunmadı.', ha='center', va='center', fontsize=20, color='green')
plt.title(f'Eksik Değer Analizi (Toplam: {row_count:,} Kayıt)', fontsize=15)
plt.tight_layout()
plt.savefig(f"{output_dir}/01_missing_values.png")

# --- 5. Zaman Serisi (Günlük/Saatlik) ---
hourly_data = df.withColumn("hour", hour("ts")).groupBy("hour").count().orderBy("hour").toPandas()
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='count', data=hourly_data, marker='o', linewidth=2.5, color='#2ecc71')
plt.title(f'Saatlik Olay Trafiği (Toplam: {row_count:,} İşlem)', fontsize=16)
plt.savefig(f"{output_dir}/02_hourly_trend.png")

daily_data = df.withColumn("date", date_format("ts", "yyyy-MM-dd")).groupBy("date").count().orderBy("date").toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x='date', y='count', data=daily_data, color='#3498db')
plt.title(f'Günlük Olay Dağılımı ({unique_users:,} Benzersiz Kullanıcı)', fontsize=16)
plt.xticks(rotation=45)
plt.savefig(f"{output_dir}/03_daily_trend.png")

# --- 6. Dağılım Analizleri ---
event_dist_pd = df.groupBy("olay_tipi").count().orderBy(desc("count")).toPandas()
plt.figure(figsize=(8, 8))
plt.pie(event_dist_pd['count'], labels=event_dist_pd['olay_tipi'], autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title(f'Olay Tipleri Dağılımı ({unique_events} Farklı Tip)', fontsize=16)
plt.savefig(f"{output_dir}/04_event_pie_chart.png")

user_activity = df.groupBy("final_id").count().toPandas()
plt.figure(figsize=(12, 6))
sns.histplot(user_activity['count'], bins=30, kde=True, color='#9b59b6')
plt.title(f'Kullanıcı Aktivite Dağılımı (Sayısal Analiz)', fontsize=16)
plt.savefig(f"{output_dir}/05_user_activity_dist.png")

print("\n" + "✅" * 20)
print(" EDA TÜM GÖRSELLERLE TAMAMLANDI ")
print("✅" * 20)
spark.stop()
