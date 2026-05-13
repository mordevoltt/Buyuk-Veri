import pandas as pd
import os
import glob
from tqdm import tqdm

def check_silver_data_full_scan():
    base_path = r"c:\Users\çağrı\Desktop\büyük veri\proje\data\delta\silver_table"
    
    # Tüm parquet dosyalarını bul
    parquet_files = glob.glob(os.path.join(base_path, "*.parquet"))
    
    if not parquet_files:
        print("Silver tabloda henüz veri dosyası oluşmamış.")
        return

    print(f"\n{'='*60}")
    print(f"SILVER TABLO (TÜM VERİ) DETAYLI DENETİMİ")
    print(f"{'='*60}")
    print(f"Toplam Dosya Sayısı: {len(parquet_files)}")

    df_list = []
    # Tüm dosyaları oku (Hız için sadece gerekli sütunları alıyoruz)
    for f in tqdm(parquet_files, desc="Dosyalar Okunuyor"):
        try:
            # Sadece kontrol edeceğimiz sütunları okumak işlemi çok hızlandırır
            temp_df = pd.read_parquet(f, columns=['timestamp', 'kullanici_id', 'olay_tipi'])
            df_list.append(temp_df)
        except Exception as e:
            continue
    
    if not df_list:
        print("Veri okunamadı.")
        return

    df = pd.concat(df_list)
    total_rows = len(df)

    print(f"\nTarama Tamamlandı!")
    print(f"-> Toplam Taranan Satır: {total_rows:,}")

    # --- Hatalı Veri Analizi ---
    # 1. Null Kontrolü
    null_count = df['kullanici_id'].isna().sum()
    
    # 2. 'Anonymous' Kontrolü
    anonymous_count = (df['kullanici_id'] == 'Anonymous').sum()

    print(f"\n{'*'*20} DENETİM SONUÇLARI {'*'*20}")
    print(f"-> NULL Kullanıcı ID Sayısı      : {null_count}")
    print(f"-> 'Anonymous' Kullanıcı Sayısı  : {anonymous_count}")
    
    total_invalid = null_count + anonymous_count

    if total_invalid == 0:
        print("\n✅ MÜKEMMEL: Tüm veri seti temiz! Filtreleme %100 başarılı.")
    else:
        fail_rate = (total_invalid / total_rows) * 100
        print(f"\n❌ DİKKAT: Toplam {total_invalid} adet hatalı satır sızmış!")
        print(f"Hata Oranı: %{fail_rate:.4f}")
        print("Öneri: Silver katmanı temizleme mantığını (spark_streaming.py) gözden geçirin.")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    check_silver_data_full_scan()
