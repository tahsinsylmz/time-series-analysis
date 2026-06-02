# Zaman Serilerinde Açıklanabilir Anomali Tespiti: Olasılıksal Otomata ile Derin Öğrenmenin Karşılaştırılması

## Özet

Bu çalışmada, çok değişkenli zaman serilerinde anomali tespiti için **beyaz kutu**
(olasılıksal sonlu otomata) ve **kara kutu** (LSTM, GRU, 1B-CNN) yaklaşımları aynı
deney protokolü altında karşılaştırılmıştır. İki ölçüt setinde — SKAB (endüstriyel
valf sensörleri) ve BATADAL (su dağıtım şebekesine siber saldırılar) — modeller
doğruluk, gürültü/dağılım kayması dayanıklılığı ve açıklanabilirlik eksenlerinde
değerlendirilmiştir. Bulgular, tek bir "en iyi" model aramaktan çok, model
davranışlarının sistematik analizine odaklanmaktadır.

## 1. Giriş

Zaman serisi anomali tespiti, endüstriyel izleme ve kritik altyapı güvenliğinde
merkezi bir problemdir. Derin öğrenme modelleri yüksek doğruluk sağlayabilir ancak
kararlarının gerekçesini gizler; oysa güvenlik açısından kritik alanlarda bir
kararın **neden** verildiği, kararın kendisi kadar önemlidir. Bu çalışma, her
kararını olasılıksal geçişler üzerinden matematiksel olarak gerekçelendiren bir
olasılıksal otomata modelini, üç derin öğrenme mimarisiyle karşılaştırarak
**doğruluk–açıklanabilirlik ödünleşimini** niceliksel olarak inceler.

## 2. İlgili Çalışmalar

Sembolik zaman serisi gösterimleri, SAX [1] ve onun temelindeki PAA [2] ile
yaygınlaşmıştır; bu yöntemler sürekli sinyali ayrık sembol dizilerine indirgeyerek
örüntü tabanlı modellemeye olanak tanır. Dizi benzerliği için Levenshtein düzenleme
mesafesi [3] klasik bir araçtır ve bu çalışmada görülmemiş örüntülerin bilinen
durumlara eşlenmesinde kullanılır. Kara kutu tarafında, LSTM [4] ve GRU [5]
yinelemeli ağları zamansal bağımlılıkları modellemede standarttır. Ölçüt setleri
olarak SKAB [6] ve BATADAL [7] kullanılmıştır.

## 3. Yöntem

### 3.1 Ön İşleme

Eksik değerler sızıntı önleyici biçimde doldurulur (SKAB'da dosya-içi, BATADAL'da
nedensel ileri-yön). `StandardScaler` ve `PCA` **yalnızca eğitim bölmesinde** uydurulur;
doğrulama/test bu dönüşümle ölçeklenir. Otomata tek boyutlu girdi istediğinden PCA'nın
birinci bileşeni (PC1) kullanılır. PC1'in açıkladığı varyans oranı SKAB'da ≈ %30,4
(8 öznitelik), BATADAL'da ≈ %20,6 (43 öznitelik) olup, BATADAL'da tek bileşene
indirgemenin bilgi kaybına yol açtığını gösterir.

### 3.2 Beyaz Kutu — Olasılıksal Otomata

Akış: PC1 → global z-normalizasyon → kayan pencere → PAA → SAX sembolizasyon →
durumlar (benzersiz SAX örüntüleri) → frekans tabanlı geçişler → Laplace
yumuşatmalı geçiş olasılıkları → yol (path) olasılığı. Bir örüntü dizisinin
olasılığı ardışık geçiş olasılıklarının çarpımıdır; düşük olasılık beklenmedik
davranışa, dolayısıyla yüksek anomali skoruna karşılık gelir:

    P(Sᵢ→Sⱼ) = (geçiş_sayısı + α) / (toplam_çıkış + α·|durumlar|),   α = 1 (Laplace)

PAA bölme faktörü 2'dir; her PAA segmenti iki ham örneğin ortalamasıdır (gerçek
boyut indirgeme). z-normalizasyon klasik SAX'taki pencere-bazlı yerine eğitim
istatistikleriyle **global** yapılır; böylece pencerelerin mutlak seviyesi korunur
(anomaliler çoğunlukla seviye/genlik sapması olduğundan kasıtlı tercih). Eğitim
sözlüğünde bulunmayan (unseen) örüntüler Levenshtein mesafesiyle en yakın bilinen
örüntüye eşlenir ve mesafe oranında ek ceza alır.

### 3.3 Kara Kutu — Derin Öğrenme

Üç mimari ortak girdi/çıktı sözleşmesini paylaşır: girdi (batch, dizi_uzunluğu,
öznitelik), çıktı tek logit. **LSTM/GRU** son zaman adımının gizli durumundan,
**1B-CNN** iki evrişim katmanı + küresel maksimum havuzlamadan ikili sınıflandırma
üretir. Eğitim parametreleri sabittir (dizi 30, epoch ≤ 50, batch 32, lr 0,001,
erken durdurma sabrı 5, gizli boyut 48); dengesizlik `pos_weight` ile telafi edilir.

### 3.4 Açıklanabilirlik Modülü

Otomata her kararı izlenebilir bileşenlere ayırır: mevcut durum, gözlemlenen örüntü,
örüntünün eğitimde bulunup bulunmadığı (`status`: seen/unseen), unseen ise eşlenen
örüntü (`mapped_to`), gerçekleşen geçişler ve olasılıkları, yol olasılığı, nihai
karar ve geçiş olasılıklarından türetilen güven skoru. Çıktı hem makine-okur (ister
X.F formatı: `time_step, state, pattern, status, mapped_to, probability, decision`)
hem insan-okur Türkçe metin olarak üretilir. Ek olarak karşıt durum (counterfactual)
analizi ve unseen örüntü mesafe özeti raporlanır.

## 4. Deney Kurulumu

- **SKAB:** dosya bazlı `StratifiedGroupKFold` (5 kat); kaynak dosya grup değişkenidir,
  böylece aynı dosyanın örnekleri eğitim ve teste sızmaz.
- **BATADAL:** zaman sıralı %60/%20/%20 bölme (geçmiş → gelecek).
- **Tohumlar:** [42, 123, 2026, 7, 999] (5 tekrar); otomata deterministik olduğundan
  kat başına bir kez eğitilir.
- **Senaryolar:** `orijinal`, `gurultu` (σ=0,3 Gauss), `unseen` (×1,8 kazanç kayması →
  sözlük-dışı örüntüler; ister VI.A). Tüm `fit` işlemleri yalnızca eğitimde yapılır.
- **İstatistik:** kat bazlı F1 için Wilcoxon işaretli sıra testi; aynı test noktaları
  için McNemar testi.

Tüm parametreler `config/config.yaml`'dan yönetilir; her koşuda kullanılan yapılandırma
`results/kullanilan_config.yaml` olarak kaydedilir (ister VIII.A deney takibi).

## 5. Sonuçlar

### 5.1 Model Karşılaştırması (doğruluk)

SKAB'da derin öğrenme açık ara öndedir (F1 ≈ 0,87–0,88), otomata ise 0,521'dir.
Otomatanın profili yüksek recall (0,972) – düşük precision (0,356) yönündedir:
anomalilerin neredeyse tümünü yakalar fakat çok yanlış alarm üretir
(bkz. `rapor/tablolar/tablo_skab_fold.md`).

BATADAL'da tablo tersine döner: bu küçük ve %5,2 dengesiz veride **otomata en yüksek
F1'e (0,100) sahiptir** (GRU 0,047, LSTM 0,040, 1B-CNN 0,000). 1B-CNN tamamen başarısız
olur; `roc_auc ≈ 0,05` değeri salt çoğunluk-sınıfı çöküşünden (≈ 0,50) düşük olduğundan
skorların **sistematik ters sıralandığını** gösterir. Sıralama ölçütünde (ROC-AUC) ise
LSTM (0,787) ve GRU (0,762) öndedir; yani DL iyi sıralar fakat tek eşikle F1'e dönüştüremez.

### 5.2 Veri Setleri Arası Fark

SKAB görece dengeli ve örüntülü olduğundan tüm modeller anlamlı F1 üretir. BATADAL
küçük, çok dengesiz ve eğitim/test dağılımı farklı olduğundan bütün modeller için
zordur. Beyaz kutu modelin BATADAL'da görece iyi durması, frekans tabanlı örüntü
modelinin az veriyle de işleyebildiğini gösterir.

**Kapsam notu (veri setleri arası transfer).** Bu çalışma SKAB ve BATADAL ile
sınırlandırılmıştır; bir veri setinde eğitip diğerinde test eden çapraz veri seti
(cross-dataset) genellenebilirlik analizi (ek istek listesindeki örnek Tablo 3,
SWAT/WADI/BATADAL) kapsam dışıdır. Gerekçe: iki veri seti **farklı öznitelik
uzaylarına** sahiptir (SKAB 8 öznitelik → PC1, BATADAL 43 öznitelik → PC1) ve
öznitelikler fiziksel olarak eşlenemez; ayrıca **etiket semantiği** farklıdır (SKAB
`anomaly`, BATADAL `ATT_FLAG`). Otomata tek boyutlu PC1 üzerinde eğitim bölmesine
uydurulan bir SAX/PAA sözlüğü ve geçiş matrisi öğrendiğinden, farklı boyutlu ve
farklı dağılımlı bir veri setine uygulanması tanımsızdır; derin öğrenme modelleri de
girdi boyutuna bağlı olduğundan doğrudan aktarılamaz. Dolayısıyla ortak bir transfer
kurgusu tanımlı olmadığından bu eksen analiz dışı bırakılmıştır (ek istek listesindeki
SWAT/WADI yalnızca biçim örneğidir).

### 5.3 Gürültü ve Dağılım Kayması Etkisi

Otomatanın F1'i gürültü ve kayma altında neredeyse sabittir (SKAB 0,521/0,522/0,512).
BATADAL'da otomata hem orijinal (0,100) hem unseen (0,122) senaryosunda tüm modeller
arasında en iyidir; görülmemiş örüntüleri Levenshtein ile ele alma yeteneği bu durumda
işe yarar (bkz. `tablo_dayaniklilik.md`, `fig_senaryo_dayaniklilik.png`).

### 5.4 Unseen Veri Davranışı

İki kavram ayrı tutulur. (1) **`unseen` senaryosu** bir *kovaryant kayma* (covariate
shift) testidir: kazanç (×1,8) ile ölçeklenmiş test setinin **tamamı** üzerinde F1
ölçülür; tablolardaki `unseen` sütunu budur. (2) **VI.A sözlük-dışı (out-of-vocabulary)
yönetimi** ise yalnızca eğitim sözlüğünde bulunmayan örüntülerin nasıl ele alındığını
ölçer ve `unseen_analizi.csv` ile *ayrı* raporlanır (bkz. Sözlük-dışı Yönetimi tablosu):
**Detection Rate** (sözlük-dışı örüntü taşıyan test konumu oranı) ve **Mapping Accuracy**
(Levenshtein ile eşlenen en yakın bilinen örüntünün, kaydırma öncesi gerçek örüntüye
doğruluğu; tam eşitlik ve mesafe≤1). Kaydırılmış SKAB test setinde gözlenen 62 sözlük-dışı
örüntünün en yakın bilinene ortalama mesafesi 1,02 (en çok 2) olup, eşlemenin küçük
düzenlemelerle çalıştığını gösterir. Bu sözlük-dışı yönetimin derin öğrenme modellerinde
kavramsal bir karşılığı yoktur (yalnızca otomataya özgüdür).

### 5.5 Parametre Etkileri

Pencere × alfabe taramasında en iyi sonuç **alfabe = 3** sütununda toplanır (F1 ≈
0,519–0,523). Pencere/alfabe büyüdükçe durum sayısı hızla artar (≈ 18 → ≈ 1157) ve
geçiş yoğunluğu düşer (≈ 0,43 → ≈ 0,004): durum uzayı büyüdükçe geçiş matrisi
seyrekleşir. F1 bu değişime büyük ölçüde duyarsızdır; model karmaşıklığını artırmak
doğruluğu iyileştirmez — sadelik lehine bir bulgu (`fig_parametre_duyarlilik.png`).

### 5.6 İstatistiksel Testler

Wilcoxon testinde otomata vs LSTM/GRU/1B-CNN için p = 0,0625 (n=5; 5 katta tutarlı
yönlü farkın ulaşabileceği en küçük değer); derin öğrenme tüm katlarda otomatadan
yüksektir. McNemar (SKAB, otomata vs 1B-CNN): istatistik = 3296,4, p ≈ 0 (yalnız
otomatanın doğru olduğu 415, yalnız 1B-CNN'in 4400 nokta); SKAB'da test geçerlidir.
**McNemar (BATADAL) yorum dışıdır:** bu veri setinde referans derin öğrenme modeli
(1B-CNN) ve diğer tüm DL modelleri dejenere olup F1 ≈ 0 üretir (her şeye "normal"
der); bu durumda ham doğruluk üzerinden kurulan McNemar tablosu dejenere modeli
kayırır ve anlamlı yorumlanamaz. Bu nedenle istatistik çıktısında BATADAL McNemar
kaydı `yorum_disi: true` bayrağıyla işaretlenir; çıkarımlar yalnızca SKAB McNemar'ı
üzerinden yapılır. **Dengesiz veride doğruluk yanıltıcıdır; asıl ölçüt F1/PR'dir.**

Görseller: karmaşıklık matrisi, ROC/PR eğrisi (`fig_roc_pr.png`; metinde anılan GRU
ROC-AUC ≈ 0,93 değeri 5 kat ortalaması, figürdeki eğri **SKAB fold-1** için AUC = 0,961),
otomata durum geçiş diyagramı ve geçiş olasılık ısı haritası `results/figurler/`
altındadır.

### 5.7 Çalışma Süreleri (EK Tablo5)

Her model için eğitim ve çıkarım (inference) süreleri tam koşu sırasında ölçülür
(`results/calisma_sureleri.csv`, `rapor/tablolar/tablo_runtime.md`). Beyaz kutu otomata,
tek boyutlu (PC1) frekans tabanlı bir model olduğundan derin öğrenme modellerine kıyasla
belirgin biçimde **daha kısa eğitim süresi** ister; çıkarım süreleri tüm modellerde küçük
ölçektedir. Bu, yorumlanabilirliğin yanında hesaplama maliyeti açısından da otomatanın
hafif olduğunu gösterir (kesin değerler için tablo_runtime.md).

## 6. Tartışma: Doğruluk–Açıklanabilirlik Ödünleşimi

SKAB'da derin öğrenme doğrulukta açık ara öndedir; ancak kararlarının gerekçesini
veremez. Otomata, her kararı durum–geçiş–olasılık zinciri olarak gösterebildiğinden
düşük doğruluğa rağmen denetlenebilirlik ve güven açısından farklı bir değer sunar.
Dengesiz BATADAL'da ise beyaz kutu model doğrulukta da öne geçebilmektedir; bu, az
veri rejiminde sade ve yorumlanabilir modellerin pratik bir alternatif olduğunu
düşündürür. Çalışmanın amacı tek bir en iyi modeli seçmek değil, model davranışlarını
bilimsel ve sistematik biçimde analiz etmektir; sonuçlar ölçüt setine ve dengeye göre
hangi yaklaşımın ne zaman tercih edilebileceğini niceliksel olarak ortaya koyar.

## 7. Sınırlılıklar ve Gelecek Çalışma

Otomatanın SKAB'daki düşük precision'ı, eşik seçiminin ve global z-normalizasyonun
birlikte yarattığı yüksek-recall rejiminden kaynaklanır. Bu rejim bir eşik *hatası*
değil, dürüst bir F1-optimal seçimin sonucudur: eşik doğrulama bölmesinde F1'i
maksimize edecek biçimde seçilir (§3) ve otomatanın SKAB'da ürettiği skorlar normal
ile anomali noktalarını yeterince ayrıştıramadığından, optimal eşik 5 kattan 3'ünde
(kat 0, 3, 4) tüm test noktalarını anomali işaretleyen bir noktada oturur (recall =
1,0, precision = sınıf taban oranı ≈ 0,34); kalan iki katta recall 0,88–0,98
aralığındadır. Uç skor adaylarını dışlayan daha tutucu bir eşik bu katlarda F1'i
düşürürdü; parametre taramasında en iyi yapılandırmanın da F1 ≈ 0,52 mertebesinde
kalması (§5.5), düşük precision'ın eşikten değil modelin SKAB'daki ayrıştırma gücünden
kaynaklandığını doğrular. İyileştirme için pencere-bazlı normalizasyon veya çok
boyutlu otomata genişlemeleri incelenebilir. BATADAL'da tek PC bileşeni
sınırlı varyans taşıdığından, otomata için çok değişkenli sembolizasyon gelecekte
denenebilir.

## Kaynakça

[1] Lin, J., Keogh, E., Lonardi, S., Chiu, B. (2003). *A Symbolic Representation of
Time Series, with Implications for Streaming Algorithms.* Proc. 8th ACM SIGMOD
Workshop on Research Issues in Data Mining and Knowledge Discovery (DMKD).

[2] Keogh, E., Chakrabarti, K., Pazzani, M., Mehrotra, S. (2001). *Dimensionality
Reduction for Fast Similarity Search in Large Time Series Databases.* Knowledge and
Information Systems, 3(3), 263–286.

[3] Levenshtein, V. I. (1966). *Binary Codes Capable of Correcting Deletions,
Insertions, and Reversals.* Soviet Physics Doklady, 10(8), 707–710.

[4] Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural
Computation, 9(8), 1735–1780.

[5] Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk,
H., Bengio, Y. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for
Statistical Machine Translation.* Proc. EMNLP.

[6] Katser, I. D., Kozitsin, V. O. (2020). *Skoltech Anomaly Benchmark (SKAB).*
Kaggle / GitHub: https://github.com/waico/SKAB

[7] Taormina, R., Galelli, S., Tippenhauer, N. O., Salomons, E., Ostfeld, A. ve diğ.
(2018). *Battle of the Attack Detection Algorithms: Disclosing Cyber Attacks on Water
Distribution Networks.* Journal of Water Resources Planning and Management, 144(8).

[8] Pedregosa, F. ve diğ. (2011). *Scikit-learn: Machine Learning in Python.* Journal
of Machine Learning Research, 12, 2825–2830.
