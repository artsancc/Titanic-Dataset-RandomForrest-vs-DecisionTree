# Titanic Hayatta Kalma Tahmini

Bu proje, ünlü Titanic veri setini kullanarak yolcuların hayatta kalma durumlarını tahmin etmek için Karar Ağaçları (Decision Tree) ve Rastgele Orman (Random Forest) algoritmalarını karşılaştırmalı olarak uygular.

# Proje Özeti

Kod, veri setini yükler, temizler, analiz eder ve iki farklı makine öğrenmesi modelini eğitir. Sonuçları sadece metriklerle değil, aynı zamanda görsel grafikler ve karar ağacı şemasıyla birlikte sunar.

# Kullanılan Teknolojiler

Python 3.x

Pandas & Seaborn: Veri manipülasyonu ve örnek veri seti yükleme.

Scikit-Learn: Model eğitimi, veri bölme ve performans metrikleri.

Matplotlib: Sonuçların görselleştirilmesi ve grafik kaydı.

# Özellikler

Veri Ön İşleme:

Eksik yaş (Age) verileri medyan değer ile doldurulur.

Cinsiyet verisi sayısal (0-1) formatına dönüştürülür.

Gereksiz sütunlar elenerek en etkili 6 özellik (Feature) seçilir.

Model Eğitimi:

DecisionTreeClassifier: 5 derinlik sınırı ile görselleştirilebilir bir ağaç oluşturur.

RandomForestClassifier: 100 ağaçtan oluşan daha karmaşık ve güçlü bir model eğitir.

Analiz ve Görselleştirme:

Karışıklık Matrisi (Confusion Matrix): Modellerin hata türlerini gösterir.

ROC Eğrisi & AUC: Modellerin ayırt edici gücünü ölçer.

Karar Ağacı Şeması: Modelin nasıl karar verdiğini adım adım (karar_agaci.png) görselleştirir.


# Kurulum ve Çalıştırma
Gerekli kütüphaneleri yükleyin:

pip install pandas seaborn matplotlib scikit-learn


# Örnek Çıktı Analizi
Program çalıştığında konsolda şu bilgileri göreceksiniz:

Veri setinin genel istatistikleri ve hayatta kalma oranları.

Her iki model için F1-Skoru, Precision ve Recall değerleri.

Hangi modelin daha yüksek doğruluk (Accuracy) oranına sahip olduğu.

## Not: 
Kod içerisinde matplotlib.use("Agg") kullanılmıştır; bu sayede kodunuz grafik arayüzü olmayan sunucularda da hata vermeden çalışır ve grafikleri doğrudan dosya olarak kaydeder.
