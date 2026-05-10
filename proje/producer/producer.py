import pandas as pd
from kafka import KafkaProducer
import json
import time
import logging
from datetime import datetime
import os

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def json_serializer(data):
    return json.dumps(data, default=str).encode('utf-8')

def start_producer():
    # Kafka ayarları - Docker içinde servis adını kullanıyoruz
    bootstrap_servers = [os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')]
    topic_name = os.getenv('KAFKA_TOPIC', 'online_retail')
    
    producer = None
    retry_count = 0
    while not producer and retry_count < 10:
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=json_serializer
            )
            logging.info("Kafka Producer başarıyla bağlandı.")
        except Exception as e:
            retry_count += 1
            logging.error(f"Kafka'ya bağlanılamadı (Deneme {retry_count}): {e}")
            time.sleep(5)
    
    if not producer:
        logging.critical("Kafka'ya bağlanılamadı, script durduruluyor.")
        return

    # CSV Dosyasından Okuma (Adım 2 Gereksinimi)
    file_path = os.getenv('DATA_FILE_PATH', '/app/data/online_retail_II.csv')
    logging.info(f"Veri yükleniyor: {file_path}")
    
    try:
        # Not: Excel dosyası varsa kullanıcı CSV'ye çevirmiş olmalı.
        df = pd.read_csv(file_path) 
        logging.info(f"Toplam {len(df)} satır okundu.")
    except Exception as e:
        logging.error(f"CSV okunurken hata oluştu: {e}")
        return

    # Mesaj gönderme hızı ayarı (Saniyede 10-100 mesaj)
    msg_per_second = int(os.getenv('MSG_PER_SECOND', 20)) 
    delay = 1.0 / msg_per_second

    logging.info(f"Veri akışı başlıyor... Hız: {msg_per_second} msg/sec")

    for index, row in df.iterrows():
        try:
            # GEREKSİNİM: timestamp, kullanici ID, olay tipi ve ilgili ID bilgileri bulunmalı
            message = {
                'timestamp': datetime.now().isoformat(),
                'kullanici ID': str(row.get('Customer ID', 'Anonymous')),
                'olay tipi': 'purchase', # Varsayılan olay tipi
                'ilgili ID': str(row.get('StockCode', 'Unknown')),
                'raw_data': row.to_dict() # Tüm satır verisini de ekliyoruz
            }
            
            # Kafka'ya gönder
            producer.send(topic_name, value=message)
            
            # Producer logları ile kaç mesaj gönderildiği takip edilebilmeli
            if (index + 1) % 100 == 0:
                logging.info(f"GÖNDERİLEN TOPLAM MESAJ: {index + 1}")
            
            # Hız kontrolü
            time.sleep(delay)
            
        except Exception as e:
            logging.error(f"Mesaj gönderilirken hata (Satır {index}): {e}")
            continue

    producer.flush()
    logging.info("Veri gönderimi tamamlandı.")

if __name__ == "__main__":
    start_producer()
