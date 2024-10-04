# Diabetes Prediction Model

Bu proje, diyabet hastalığını tahmin etmek için veri analizi ve makine öğrenimi yöntemlerini kullanarak bir model oluşturmayı hedefler. Veriler, diyabet teşhisi için önemli olan çeşitli özellikleri içeren bir veri setinden alınmıştır.

## Proje İçeriği

Bu proje aşağıdaki adımları içermektedir:

1. **Veri Seti İncelemesi**
   - Veri setinin genel yapısı ve içeriği incelenmiştir.
   - Eksik değer analizi ve temel istatistikler sağlanmıştır.

2. **Yeni Değişkenler Oluşturma**
   - BMI (Vücut Kitle İndeksi) kategorileri oluşturulmuştur.
  
3. **Encoding İşlemleri**
   - Kategorik değişkenler sayısal verilere dönüştürülmüştür.
   - One-Hot Encoding yöntemi kullanılmıştır.

4. **Standartlaştırma**
   - Numerik değişkenler için `StandardScaler` kullanılarak veriler standartlaştırılmıştır.

5. **Model Oluşturma**
   - `Logistic Regression` modeli oluşturulmuş ve sonuçlar değerlendirilmiştir.
   - Modelin performansı, karmaşıklık matrisleri ve sınıflandırma raporları ile analiz edilmiştir.

## Kullanım

Projeyi yerel ortamınıza klonlayarak çalıştırabilirsiniz:

```bash
git clone https://github.com/EceDalpolat/diabetesCSVForFeatureEngineering.git
cd repo_adi
