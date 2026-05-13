# TEKNİK RAPOR: Büyük Veri Analitiği ve Makine Öğrenmesi Hattı Projesi

## 1. Proje Özeti ve Kapsamı
Bu proje, modern bir büyük veri mimarisi kullanarak uçtan uca bir veri işleme ve makine öğrenmesi (ML) boru hattı (pipeline) oluşturmayı hedeflemektedir. Proje, gerçek zamanlı veri akışından (streaming) başlayarak, verinin temizlenmesi, zenginleştirilmesi, Delta Lake üzerinde depolanması ve nihayetinde gelişmiş regresyon modelleri ile tahminleme yapılması süreçlerini kapsar. 

Projenin temel amacı, bir perakende veri setini (Online Retail II) kullanarak müşteri harcamalarını tahmin edebilecek ölçeklenebilir ve güvenilir bir sistem inşa etmektir.

---

## 2. Proje Mimarisi (Architecture)
Proje, mikroservis mimarisine dayalı, Docker konteynerleri üzerinde koşan entegre bir yapıya sahiptir.

### 2.1. Altyapı ve Bileşenler
- **Konteynerleştirme (Docker & Docker Compose):** Tüm sistem bileşenleri (Kafka, Spark, MLflow, Dashboard) Docker üzerinde izole edilmiştir. Bu, sistemin farklı ortamlarda tutarlı çalışmasını sağlar.
- **Veri Akış Katmanı (Kafka & Zookeeper):** Ham verilerin yüksek hızda ve güvenilir bir şekilde iletilmesini sağlar. Python tabanlı bir `producer`, CSV verilerini JSON formatına dönüştürerek Kafka kanallarına besler.
- **İşleme Motoru (Apache Spark):** Büyük veri işleme kapasitesi için Spark tercih edilmiştir. Spark Streaming, Kafka'dan gelen verileri gerçek zamanlı olarak tüketir.
- **Depolama Katmanı (Delta Lake):** Veri ambarı mimarisinde "Medallion" (Madalyon) yapısı kullanılmıştır:
    - **Bronze (Ham):** Kafka'dan gelen ham veriler meta-verileriyle birlikte saklanır.
    - **Silver (Temizlenmiş):** Boş değerlerin filtrelendiği, tiplerin dönüştürüldüğü ve duplikelerin temizlendiği katmandır.
    - **Gold (Analitik):** ML modelleri için özelliklerin (features) hazırlandığı ve agregasyonların yapıldığı katmandır.
- **Deney Takibi ve Model Yönetimi (MLflow):** Model eğitim süreçleri, parametreler, metrikler ve görsel çıktılar MLflow üzerinde kayıt altına alınır.
- **Görselleştirme (Streamlit):** Model performansları ve veri analizleri interaktif bir dashboard üzerinden sunulur.

---

## 3. Uygulama Detayları

### 3.1. Veri İşleme Hattı
Spark Streaming kullanılarak kurulan hatta, veriler saniyede 10-100 mesaj hızında işlenir. **Watermarking** teknolojisi kullanılarak geciken veriler yönetilir ve sistemin sürekliliği sağlanır. Delta Lake'in "Time Travel" özelliği sayesinde verinin geçmiş sürümlerine erişim ve tutarlılık (ACID) garantilenmiştir.

### 3.2. Özellik Mühendisliği (Feature Engineering)
Ham veriden anlamlı çıkarımlar yapmak için şu teknikler uygulanmıştır:
- **Log Dönüşümü:** Hedef değişken olan `total_spending` verisindeki çarpıklığı gidermek için logaritmik dönüşüm uygulanmıştır.
- **Aykırı Değer (Outlier) Yönetimi:** `approxQuantile` yöntemi ile %1 ve %99 dilimleri dışındaki veriler "clipping" yöntemiyle tıraşlanmıştır.
- **Yeni Özellikler:** Müşteri sadakat skoru (loyalty score), günlük ortalama aktivite ve müşteri değer skoru gibi RFM (Recency, Frequency, Monetary) temelli özellikler türetilmiştir.
- **Polinom Açılımı:** Değişkenler arasındaki etkileşimleri yakalamak için 2. dereceden polinom özellikleri eklenmiştir.

### 3.3. Makine Öğrenmesi Modelleri
Proje kapsamında 5 farklı regresyon algoritması eğitilmiş ve karşılaştırılmıştır:
1. **Linear Regression (Lasso/Ridge):** Temel tahminleme ve katsayı analizi.
2. **Decision Tree Regressor:** Doğrusal olmayan ilişkilerin yakalanması.
3. **Random Forest Regressor:** Topluluk (ensemble) öğrenmesi ile daha kararlı sonuçlar.
4. **Gradient Boosted Trees (GBT):** Hata odaklı ardışık öğrenme.
5. **Generalized Linear Regression (GLR):** Farklı dağılım aileleri için esneklik.

**Optimizasyon:** Tree tabanlı modellerde **Optuna** kütüphanesi kullanılarak hiperparametre optimizasyonu yapılmış, en iyi modeller 3-katlı çapraz doğrulama (3-fold Cross Validation) ile seçilmiştir.

---

## 4. Karşılaşılan Zorluklar ve Çözümler

1. **Docker Ağ Bağlantıları (Networking):** Kafka, Spark ve MLflow konteynerlerinin birbirleriyle düşük gecikmeli iletişim kurması için özel bir `bigdata-network` köprüsü kurulmuştur.
2. **Streaming Veri Tutarlılığı:** Akış verisinde mükerrer kayıtların (duplicate) oluşması, Spark'ın `dropDuplicates` ve `watermarking` özellikleri ile çözülmüştür.
3. **Model Ölçeklenebilirliği:** Eğitim setinin büyümesi durumunda `RobustScaler` ve vektörleştirme işlemleri Spark'ın dağıtık yapısı sayesinde donanım kısıtına takılmadan gerçekleştirilmiştir.
4. **MLflow Entegrasyonu:** Farklı model tiplerinden (Pipeline vs Single Model) gelen metriklerin ve artifact'lerin tutarlı bir şekilde loglanması için özel sarmalayıcı (wrapper) fonksiyonlar geliştirilmiştir.

---

## 5. Sonuçlar ve Değerlendirme

### 5.1. Model Performansları
Eğitilen modeller R2, RMSE ve MAE metriklerine göre değerlendirilmiştir. Özellikle **Random Forest** ve **GBT** modellerinin, karmaşık müşteri davranışlarını yakalamada doğrusal modellere göre daha yüksek başarı gösterdiği gözlemlenmiştir. 
- Logaritmik uzaydaki RMSE değerleri, modellerin genel hata trendini gösterirken;
- Gerçek para birimi bazındaki RMSE ve MAE değerleri, iş birimleri için anlamlı finansal içgörüler sunmaktadır.

### 5.2. Sistem Verimliliği
- **Gecikme (Latency):** Verinin Kafka'ya girişinden Gold tabloya yazılmasına kadar geçen süre saniyeler mertebesindedir.
- **Görselleştirme:** Streamlit dashboard'u sayesinde, MLflow'dan çekilen anlık verilerle model performansları ve "Feature Importance" (Özellik Önem Derecesi) grafikleri kolayca analiz edilebilmektedir.

### 5.3. Gelecek Çalışmalar
Sistemin başarısını artırmak için şu adımlar atılabilir:
- **Derin Öğrenme:** PySpark ML yerine Horovod veya Petastorm kullanılarak derin öğrenme (Deep Learning) entegrasyonu.
- **A/B Testleri:** Canlı ortamda farklı modellerin performansını karşılaştırmak için model sunucu (Model Serving) katmanının eklenmesi.
- **NLP Entegrasyonu:** Ürün açıklamaları üzerinde doğal dil işleme yapılarak daha zengin özellikler türetilmesi.

---

## 6. Sonuç
Bu çalışma, büyük veri ekosistemindeki modern araçların birbirleriyle nasıl uyumlu çalışabileceğini kanıtlayan kapsamlı bir projedir. Veri akışından interaktif sonuç sunumuna kadar olan süreç, endüstri standartlarında (ACID uyumlu Delta Lake, MLflow deney takibi) inşa edilmiştir. Sistem, yüksek veri hacimlerini işleyebilecek esneklikte ve doğrulukta bir altyapı sunmaktadır.

**Hazırlayan:** Antigravity AI (Asistan)
**Tarih:** 13 Mayıs 2026
