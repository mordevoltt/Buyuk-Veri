# Büyük Veri Analizine Giriş Dönem Projesi: Online Retail II

## Proje Hakkında
Bu proje, "Online Retail II" veri setini kullanarak uçtan uca bir büyük veri işleme ve makine öğrenmesi boru hattı (pipeline) oluşturmayı amaçlar.

## Mimari Yapı
1. **Veri Üretimi:** Kafka Producer ile simüle edilen gerçek zamanlı veri akışı.
2. **Mesajlaşma:** Zookeeper ve Kafka (Docker tabanlı).
3. **Akış İşleme:** Spark Structured Streaming ile Bronze, Silver ve Gold Delta Lake tabloları.
4. **Analiz:** Kapsamlı Keşifsel Veri Analizi (EDA).
5. **Özellik Mühendisliği:** Zaman bazlı ve kullanıcı odaklı özellik türetimi.
6. **Makine Öğrenmesi:** 5 farklı regresyon modeli, Optuna hiperparametre optimizasyonu ve MLflow takibi.
7. **Dashboard:** Streamlit tabanlı interaktif görselleştirme arayüzü.

## Adımlar
- **Adım 1:** Docker Altyapısı
- **Adım 2:** Kafka Veri Akışı
- **Adım 3:** Spark Streaming & Delta Lake
- **Adım 4:** Keşifsel Veri Analizi (EDA)
- **Adım 5:** Özellik Mühendisliği
- **Adım 6:** Makine Öğrenmesi ve MLflow
- **Adım 7:** Görselleştirme ve Dashboard
