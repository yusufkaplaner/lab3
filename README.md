# 📊 Logistic vs Linear Regression Performance Lab

Bu proje, **Social Network Ads** veri seti üzerinde Logistic Regression ve Linear Regression modellerinin performansını karşılaştırmak amacıyla geliştirilmiştir.  
Sınıflandırma problemlerinde hangi modelin neden daha üstün olduğunu hem metriklerle hem de görselleştirmelerle ortaya koyar.

---

## 🚀 Proje Amacı
- Aynı veri seti üzerinde iki farklı regresyon türünü eğiterek;
  - Doğruluk (**Accuracy**), Keskinlik (**Precision**), Duyarlılık (**Recall**) ve **F1-Score** değerlerini karşılaştırmak  
  - Linear Regression'ın bir sınıflandırma problemi için neden *teorik olarak* uygun olmadığını kanıtlamak  
  - Karar sınırlarını (**Decision Boundaries**) görselleştirerek modellerin çalışma mantığını analiz etmek  

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
- **Python 3.x**
- **Pandas & NumPy** → Veri manipülasyonu ve işleme  
- **Scikit-Learn** → Model eğitimi, ölçeklendirme ve metrik hesaplama  
- **Matplotlib** → Karar sınırlarının ve sonuçların görselleştirilmesi  

---

## 📈 Uygulama Adımları
1. **Veri Ön İşleme (Preprocessing)**  
   - `User ID` ve `Gender` gibi model performansına etkisi olmayan sütunlar drop edildi.  
   - `Age` ve `EstimatedSalary` bağımsız değişken ($X$), `Purchased` ise hedef ($y$) olarak belirlendi.  
   - **Feature Scaling**: Değerler arasındaki büyük farkları dengelemek için `StandardScaler` uygulandı.  

2. **Model Eğitimleri**  
   - **Logistic Regression**: Sınıflandırma için sigmoid fonksiyonu kullanılarak eğitildi.  
   - **Linear Regression**: Sürekli çıktı üreten model eğitildi ve `0.5` eşik değeri (threshold) ile sınıflandırmaya zorlandı.  

3. **Karşılaştırma Analizi**  
   - **Lineer Model Problemi**: Linear Regression çıktıları `[0,1]` dışına çıkabilir, bu da olasılık yorumunu imkansız kılar.  
   - **Doğruluk**: Veri seti doğrusal ayrılabilir olduğu için metrikler yakın çıksa da, Logistic Regression her zaman daha stabil bir yapı sunar.  

---

## 📊 Görselleştirme
Proje sonunda iki modelin **Decision Boundary (Karar Sınırı)** farklarını gösteren ve **Confusion Matrix (Karmaşıklık Matrisi)** analizlerini içeren grafikler oluşturulur.

---

## 📂 Dosya Yapısı 

---

## 💻 Nasıl Çalıştırılır?
Repoyu klonlayın:
```bash
git clone https://github.com/kullaniciadi/proje-adi.git
pip install pandas numpy matplotlib scikit-learn
python main.py

Bu haliyle GitHub’da README dosyan çok daha düzenli, profesyonel ve okunabilir olacak. İstersen sana ayrıca **Confusion Matrix** ve **Decision Boundary** görsellerini nasıl ekleyeceğini de gösterebilirim. Onları da README’ye dahil etmek ister misin?
