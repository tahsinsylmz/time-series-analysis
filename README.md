# Kara Kutudan Açıklanabilirliğe: Zaman Serisi Analizi için Olasılıksal Otomatalar

Zaman serisi anomali tespitinde **yorumlanabilir olasılıksal otomata** ile
**kara kutu derin öğrenme** modellerini (LSTM, GRU, 1D-CNN) aynı deney
düzeneğinde karşılaştıran bir çalışmadır. İki gerçek endüstriyel veri seti
(SKAB ve BATADAL) üzerinde doğruluk, gürültü/dağılım kayması dayanıklılığı ve
**açıklanabilirlik** ekseninde bir değerlendirme yapılır.

> Temel soru: Bir modelin yüksek doğruluğu mu yoksa her kararını *neden* verdiğini
> gösterebilmesi mi daha değerli? Bu projede ödünleşim (trade-off) somut sayılarla
> gösterilmektedir.

---

## 1. Motivasyon

Derin öğrenme modelleri zaman serisi anomali tespitinde güçlüdür ama bir
kararın *neden* alındığını gizler. Endüstriyel/güvenlik kritik sistemlerde
"bu nokta neden anomali?" sorusunun yanıtlanabilmesi gerekir. Olasılıksal
otomata, her kararı izlenebilir bileşenlere (durum, geçiş, olasılık, görülmemiş
örüntü cezası) ayırarak **beyaz kutu** bir alternatif sunar. Bu çalışma iki
yaklaşımı dürüst ve sızıntısız bir düzenekte karşılaştırır.

## 2. Veri Setleri

| Veri seti | Satır | Öznitelik | Anomali oranı | Bölme stratejisi |
|-----------|------:|----------:|--------------:|------------------|
| **SKAB** (valve1 + valve2) | 22.472 | 8 | %34,8 | Dosya bazlı `StratifiedGroupKFold` (5 kat) |
| **BATADAL** (dataset04) | 4.177 | 43 | %5,2 | Zaman sıralı %60 / %20 / %20 |

- **SKAB**: <https://github.com/waico/SKAB> — valf test tezgâhında sensör arızaları.
- **BATADAL**: <https://www.batadal.net/data.html> — su dağıtım şebekesine siber saldırılar.
- **Hedef (etiket) sütunları:** SKAB'da `anomaly`, BATADAL'da **`ATT_FLAG`** (`1` = saldırı/anomali → pozitif sınıf; `-999` = normal → `0`). Zaman sütunları (`datetime` / `DATETIME`) ve SKAB'daki `changepoint` / `source_group` / `source_file` yalnızca veri takibi ve dosya bazlı bölme için kullanılır; model girdisine alınmaz.

Veriyi yerinde doğrulamak için: `python -m scripts.download_data`

## 3. Yöntem

### 3.1 Ön İşleme (sızıntı önleme)

Tüm `fit` işlemleri **yalnızca eğitim** bölmesinde yapılır:
`StandardScaler` ve `PCA` eğitim verisine uydurulur, doğrulama/test bu
dönüşümle ölçeklenir. Eksik değerler zaman sıralı doğrusal interpolasyonla
doldurulur. Otomata tek boyutlu girdi istediğinden PCA'nın birinci bileşeni
(**PC1**) kullanılır; derin öğrenme modelleri çok değişkenli ölçeklenmiş
seriyi alır. PC1'in açıkladığı varyans oranı (eğitim bölmesinde): **SKAB ≈ %30,4**
(8 özellik, katlar arası ortalama), **BATADAL ≈ %20,6** (43 özellik). BATADAL'da
tek bileşenin payı düşüktür; bu, otomatanın tek boyuta indirgemesinin BATADAL'da
bilgi kaybına yol açtığını ve düşük başarıya katkıda bulunduğunu gösterir.

### 3.2 Beyaz Kutu — Olasılıksal Otomata

```
PC1 → z-normalizasyon → kayan pencere → PAA → SAX sembolizasyon
    → durumlar (SAX örüntüleri) → frekans tabanlı geçişler
    → Laplace yumuşatmalı geçiş olasılıkları → yol (path) olasılığı
```

- **PAA** (Piecewise Aggregate Approximation): pencereyi segment ortalamalarına indirger.
  Ham pencere uzunluğu `window_size × paa_bolme_faktoru`'dur; `paa_bolme_faktoru = 2` ile
  her PAA segmenti 2 ham örneğin ortalamasıdır (gürültüyü yumuşatır, durum sayısını azaltır).
- **SAX** (Symbolic Aggregate approXimation): Gauss kesim noktalarıyla sembole çevirir.
  z-normalizasyon klasik SAX'taki gibi pencere-bazlı değil, eğitim istatistikleriyle
  **global** yapılır; böylece pencerelerin mutlak seviyesi korunur (anomaliler çoğunlukla
  seviye/genlik sapması olduğundan kasıtlı tercih — pencere-bazlı z-norm ile F1 aynı çıkar).
- Her benzersiz SAX örüntüsü bir **durum**; durumlar arası geçişler frekansla öğrenilir:
  `P(Sᵢ→Sⱼ) = (sayım + α) / (toplam_çıkış + α·|durumlar|)` (Laplace, α=1).
- Bir örüntü dizisinin **yol olasılığı** ardışık geçiş olasılıklarının çarpımıdır;
  düşük olasılık → beklenmedik gidişat → yüksek anomali skoru.
- **Görülmemiş (unseen) örüntü**: eğitim sözlüğünde yoksa **Levenshtein** düzenleme
  mesafesiyle en yakın bilinen örüntüye eşlenir ve mesafe oranında ceza eklenir.

### 3.3 Kara Kutu — Derin Öğrenme

Üç mimari aynı pencereleme (uzunluk 30, adım 3, etiket = pencerenin son adımı)
ve aynı eğitim altyapısıyla kullanılır:

- **LSTM**, **GRU**: son gizli durum → dropout → doğrusal katman.
- **1D-CNN**: iki evrişim katmanı → küresel maksimum havuzlama → doğrusal katman.
- Dengesiz veri için `BCEWithLogitsLoss(pos_weight)`, doğrulama kaybına göre
  erken durdurma, en iyi ağırlıkların geri yüklenmesi.

### 3.4 Açıklanabilirlik

Otomatanın her kararı, makine-okur (JSON) ve insan-okur (Türkçe metin) olarak
gerekçelendirilir: pencere/PAA/SAX değerleri, son geçişler ve olasılıkları, yol
olasılığı, görülmemiş örüntüler + Levenshtein mesafesi, nihai skor/eşik/karar ve
bir güven skoru. Örnek için: `python -m scripts.demo_explain`

## 4. Deney Kurulumu

- **Senaryolar (test anında):** model temiz veriyle bir kez eğitilir, üç koşulda
  değerlendirilir — `orijinal`, `gurultu` (normalize veriye Gauss gürültüsü, σ=0,3),
  `unseen` (×1,8 kazanç kayması → eğitimde görülmemiş örüntüler / dağılım kayması).
  `unseen` senaryosu doğrudan **VI.A** tanımını gerçekler: eğitimden çıkarılan SAX
  sözlüğünde **bulunmayan** örüntüler kazanç kaymasıyla ortaya çıkar; bu örüntüler
  Levenshtein ile en yakın bilinen duruma eşlenir (bkz. §3.2) ve her koşuda kaç adet
  sözlük-dışı (novel) örüntü gözlendiği günlüğe yazılır. Senaryonun dayanıklılık
  metrikleri (F1 vb.) kaydırılmış test setinin tamamında ölçülür.
- **Tekrarlanabilirlik:** 5 rastgele tohum `[42, 123, 2026, 7, 999]`; otomata
  deterministik olduğundan kat başına bir kez eğitilir.
- **İstatistik:** kat bazlı F1 için **Wilcoxon** işaretli sıra testi; aynı test
  noktalarındaki kararlar için **McNemar** testi.
- Tüm ayarlanabilir parametreler `config/config.yaml` dosyasından yönetilir; kodda
  deney davranışını etkileyen sabit sayı yoktur (tek istisna, sıfıra bölmeyi önleyen
  `1e-8`/`1e-12` gibi matematiksel epsilon sabitleridir).

## 5. Sonuçlar

> Aşağıdaki tüm sayılar `results/` altındaki gerçek deney çıktılarından alınmıştır
> (5 kat × 5 tohum × 3 model + otomata). Yeniden üretmek için: `python -m scripts.run_experiments`

### 5.1 Doğruluk (orijinal senaryo, F1 ortalama ± standart sapma)

**SKAB**

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Otomata | 0,372 | 0,356 | **0,972** | 0,521 ± 0,009 |
| LSTM | 0,916 | 0,940 | 0,828 | 0,875 ± 0,054 |
| GRU | 0,916 | 0,944 | 0,825 | 0,874 ± 0,056 |
| 1D-CNN | 0,921 | 0,950 | 0,831 | **0,882 ± 0,048** |

**BATADAL**

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Otomata | 0,782 | 0,083 | 0,125 | **0,100** |
| LSTM | 0,892 | 0,062 | 0,030 | 0,040 ± 0,080 |
| GRU | 0,877 | 0,065 | 0,037 | 0,047 ± 0,061 |
| 1D-CNN | 0,888 | 0,000 | 0,000 | 0,000 ± 0,000 |

![Model F1 karşılaştırması](results/figurler/fig_f1_karsilastirma.png)

- SKAB'da derin öğrenme açık ara önde (F1 ≈ 0,87–0,88 vs otomata 0,52). Otomata
  **yüksek recall (0,97) – düşük precision (0,36)** profiline sahip: anomalilerin
  neredeyse tümünü yakalar ama çok yanlış alarm üretir.
- BATADAL herkes için zordur (küçük, çok dengesiz, eğitim/test dağılımı farklı).
  **1D-CNN F1 = 0'a düşer (seçilen eşikle hiç anomali işaretlemez)**; üstelik
  `roc_auc ≈ 0,05` (5 tohum ort.) salt çoğunluk-sınıfı çöküşünden (≈ 0,50)
  belirgin biçimde düşüktür — yani skorlar **sistematik olarak ters sıralanmış**:
  ağ gerçek anomalilere düşük skor verir (dağılım kayması altında dejenere eğitim).
  İlginç biçimde **otomata BATADAL orijinal senaryosunda en yüksek F1'e (0,100)
  sahiptir** (GRU 0,047, LSTM 0,040, 1D-CNN 0,000) — beyaz kutu model bu küçük ve
  dengesiz veride az da olsa gerçek anomali yakalar. (ROC-AUC'de ise LSTM/GRU
  0,76–0,79 ile öndedir; yani sıralama iyi fakat tek eşikle F1'e dönüşmüyor.)

### 5.2 Dayanıklılık (senaryolar arası F1)

| Veri | Model | Orijinal | Gürültü | Unseen |
|------|-------|---------:|--------:|-------:|
| SKAB | Otomata | 0,521 | 0,522 | 0,512 |
| SKAB | GRU | 0,874 | 0,859 | 0,877 |
| SKAB | 1D-CNN | 0,882 | 0,862 | 0,878 |
| BATADAL | Otomata | **0,100** | 0,159 | **0,122** |
| BATADAL | 1D-CNN | 0,000 | 0,000 | 0,000 |

![Senaryo dayanıklılığı](results/figurler/fig_senaryo_dayaniklilik.png)

- Otomatanın F1'i gürültü ve dağılım kayması altında neredeyse **sabit** kalır
  (SKAB'da 0,51–0,52). Bu, frekans tabanlı örüntü modelinin küçük bozulmalara
  dayanıklı olduğunu gösterir.
- BATADAL'da otomata hem orijinal (0,100) hem `unseen` (0,122) senaryosunda **tüm
  modeller arasında en iyidir** — görülmemiş örüntüleri Levenshtein ile ele alma
  yeteneği bu küçük/dengesiz veride işe yarar.

### 5.3 Parametre Duyarlılığı (otomata, SKAB)

Pencere boyutu × alfabe boyutu taraması; en iyi sonuç **alfabe = 3** sütununda
toplanır (window 3–6 arası F1 ≈ 0,519–0,523; pratikte stabil). Figürde üç metrik
birlikte gösterilir: **F1**, **durum sayısı** ve **geçiş yoğunluğu**.

- **Durum sayısı**, pencere ve alfabe boyutu büyüdükçe hızla artar (w=3,a=3'te
  ≈ 18 durumdan w=6,a=6'da ≈ 1157 duruma): daha uzun/zengin örüntüler daha çok
  benzersiz SAX kelimesi üretir.
- **Geçiş yoğunluğu** (gözlenen geçişlerin olası geçişlere oranı) ters yönde,
  ≈ 0,43'ten ≈ 0,004'e düşer: durum uzayı büyüdükçe geçiş matrisi **seyrekleşir**.
- F1 ise bu değişime büyük ölçüde **duyarsız** kalır (≈ 0,52); model karmaşıklığını
  artırmak SKAB'da doğruluğu iyileştirmez — sadelik lehine bir bulgu.

![Parametre duyarlılığı](results/figurler/fig_parametre_duyarlilik.png)

### 5.4 İstatistiksel Testler

- **Wilcoxon (SKAB, kat bazlı F1):** otomata vs LSTM/GRU/1D-CNN için p = 0,0625
  (n = 5; bu, 5 katta tutarlı yönlü farkın ulaşabileceği en küçük p değeridir).
  Derin öğrenme tüm katlarda otomatadan yüksektir.
- **McNemar (SKAB, otomata vs 1D-CNN):** istatistik = 3296,4, p ≈ 0; yalnız
  otomatanın doğru olduğu 415 nokta, yalnız 1D-CNN'in doğru olduğu 4400 nokta —
  SKAB'da derin öğrenme net üstün.
- **McNemar (BATADAL):** istatistik = 25,3, p = 4,9·10⁻⁷. Burada ham *doğruluk*
  her şeye "normal" diyen 1D-CNN'i kayırır; oysa F1 otomatanın az da olsa gerçek
  anomali yakaladığını, 1D-CNN'in hiç yakalamadığını gösterir. **Dengesiz veride
  accuracy yanıltıcıdır; asıl ölçüt F1/PR'dir.**

![Karmaşıklık matrisi](results/figurler/fig_karmasiklik_matrisi.png)

SKAB'da en iyi derin öğrenme modelinin (GRU) ROC ve Precision-Recall eğrileri
(ROC-AUC ≈ 0,93); eşikten bağımsız sıralama başarısını gösterir:

![ROC ve PR eğrileri](results/figurler/fig_roc_pr.png)

## 6. Açıklanabilirlik Örneği

`scripts/demo_explain.py` çıktısından **doğru yakalanmış** gerçek bir anomali
(SKAB ilk kat, konum 3029, gerçek etiket = 1 → doğru pozitif):

```
Durum (SAX) : aaaa
Yol         : aaaa -> baab -> aaaa
Skor / Eşik : 9.049 / 0.126  ->  Karar: ANOMALİ (güven 0.92), gerçek etiket = 1
Açıklama    : Pencere sonu durumu 'aaaa'. Son 2 geçişin yol olasılığı 1.176e-04
              (düşük olasılık = beklenmedik gidişat). Toplam skor 9.05 eşiği (0.13)
              aştığı için ANOMALİ.
```

Demo seti tek bir uca bağlı kalmaz: en yüksek skorlu noktalar, yukarıdaki gibi bir
**doğru pozitif** ve en düşük skorlu nokta birlikte yazılır (her örnekte `kategori`
alanı vardır). Otomata yüksek-recall/düşük-precision çalıştığından (precision ≈ 0,34)
en yüksek skorlu noktalar çoğu zaman **yanlış pozitiftir**; bu örnek seti hem doğru
yakalamayı hem de bu eğilimi şeffaf gösterir. Görülmemiş örüntülerde devreye giren
Levenshtein cezası ise `unseen` senaryosunda gözlenir (bkz. §3.2).

Derin öğrenme modelleri bu tür bir gerekçe **üretemez**; projenin temel katkısı
budur. Otomatanın iç yapısı da görselleştirilebilir:

| Geçiş olasılık matrisi | Durum geçiş diyagramı |
|---|---|
| ![](results/figurler/fig_gecis_matrisi.png) | ![](results/figurler/fig_durum_diyagrami.png) |

## 7. Kurulum ve Çalıştırma

```bash
# 1) Bağımlılıklar
pip install -r requirements.txt

# 2) Veri kontrolü (data/ altında SKAB ve BATADAL hazır olmalı)
python -m scripts.download_data

# 3) Birim testler
pytest -q

# 4) Tüm deneyler (5 kat × 5 tohum + parametre taraması, ~9 dk)
python -m scripts.run_experiments
#    Hızlı duman testi için:  python -m scripts.run_experiments --hizli

# 5) Figürler ve açıklanabilirlik demosu
python -m scripts.make_figures
python -m scripts.demo_explain
```

Çıktılar `results/` altına yazılır: `olcumler.csv`, `ozet.csv`,
`istatistik_testleri.json`, `parametre_taramasi_skab.csv`, `figurler/`, `aciklamalar/`.

## 8. Proje Yapısı

```
config/config.yaml          Tüm parametreler (tek kaynak)
src/
  preprocessing/            Veri yükleme, ölçekleme/PCA, bölme, gürültü
  models/
    base.py                 AnomaliModeli arayüzü, ModelGirdisi
    automata/               PAA, SAX, Levenshtein, otomata, otomata modeli
    deep_learning/          LSTM/GRU/1D-CNN ağları, veri kümesi, eğitici, model
  explainability/           Karar açıklayıcı (JSON + Türkçe metin)
  experiments/              Metrikler, istatistik testleri, senaryolar, koşturucu
  utils/                    Konfig, segmentler, eşik, tohumlama
scripts/                    run_experiments, make_figures, demo_explain, download_data
tests/                      Birim testler (Levenshtein, PAA, SAX, otomata)
results/                    Deney çıktıları, figürler, açıklamalar
```

## 9. Tasarım İlkeleri

- **SOLID/OOP:** koşturucu somut modellere değil `AnomaliModeli` arayüzüne bağımlıdır;
  yeni model eklemek için yalnızca fabrika genişletilir (açık/kapalı ilkesi).
- **Sızıntı yok:** tüm `fit` işlemleri yalnızca eğitim bölmesinde.
- **Tekrarlanabilirlik:** sabit tohumlar, merkezî konfig; kodda yalnızca matematiksel
  epsilon sabitleri kalır, ayarlanabilir parametreler config'tedir.
- **Dürüst sonuç:** tüm tablolar gerçek deney çıktısıdır; yer tutucu/uydurma değer yoktur.

## 10. Özet Bulgu

Doğrulukta derin öğrenme (özellikle SKAB'da) üstündür; ancak otomata **gürültü ve
dağılım kaymasına dayanıklı**, **görülmemiş örüntülerde rekabetçi** ve en önemlisi
**her kararı açıklanabilir**dir. Çalışma, *yorumlanabilirlik–doğruluk ödünleşimini*
gerçek veriyle somut biçimde ortaya koyar.
