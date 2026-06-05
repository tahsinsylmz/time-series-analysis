---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
Bu dosya Marp ile slayta donusur:
  - VS Code "Marp for VS Code" eklentisi ile onizleme/Export (PPTX, PDF)
  - veya CLI: npx @marp-team/marp-cli rapor/sunum.md -o sunum.pptx
Slaytlar "---" ile ayrilir; <!-- ... --> bloklari konusmaci notudur.
-->

# Kara Kutudan Açıklanabilirliğe
## Zaman Serisi Analizinde Olasılıksal Otomatalar

**Yazılım Geliştirme — 2. Proje**

Ahmet Tahsin Söylemez & Murat Öztürk

GitHub: `github.com/tahsinsylmz/time-series-analysis`

<!--
Açılış: Projede zaman serisi anomali tespitinde iki paradigmayı KARŞILAŞTIRIYORUZ —
yorumlanabilir otomata vs kara kutu derin öğrenme. Amaç en iyi modeli seçmek değil,
davranışlarını sistematik analiz etmek.
-->

---

## Problem ve Motivasyon

- Zaman serisi anomali tespiti: endüstriyel izleme, kritik altyapı güvenliği.
- **Derin öğrenme** yüksek doğruluk verir ama kararının *gerekçesini gizler.*
- Güvenlik-kritik alanlarda "bu nokta **neden** anomali?" sorusu, kararın kendisi kadar önemli.
- **Olasılıksal otomata**: her kararı durum–geçiş–olasılık zinciri olarak gösteren **beyaz kutu** alternatif.

> Temel soru: yüksek doğruluk mu, yoksa kararı açıklayabilmek mi? → **doğruluk–açıklanabilirlik ödünleşimi**

---

## Araştırma Sorusu ve Amaç

**Farklı modelleme yaklaşımları, farklı veri koşulları altında nasıl davranır ve bu fark istatistiksel olarak anlamlı mıdır?**

Hedefler:
- Karşılaştırmalı analiz (beyaz kutu vs kara kutu)
- Performansın veri setine bağımlılığı
- Gürültü ve görülmemiş (unseen) veri davranışı
- Açıklanabilirlik analizi

---

## Veri Setleri

| Veri seti | Satır | Öznitelik | Anomali | Bölme |
|-----------|------:|----------:|--------:|-------|
| **SKAB** (valve1+valve2) | 22.472 | 8 | %34,8 | Dosya bazlı `StratifiedGroupKFold` (5 kat) |
| **BATADAL** (dataset04) | 4.177 | 43 | %5,2 | Zaman sıralı %60/%20/%20 |

- SKAB: valf test tezgâhında sensör arızaları.
- BATADAL: su dağıtım şebekesine siber saldırılar (etiket: `ATT_FLAG`).
- Meta sütunlar (`datetime`, `source_file`...) model girdisine **alınmaz**.

<!--
SKAB dengeli ve örüntülü; BATADAL küçük, çok dengesiz ve dağılım kayması içeriyor.
Bu fark sonuçların yorumunda kritik olacak.
-->

---

## Genel Mimari (Pipeline)

```
Ham seri
  └─ Ön işleme: eksik veri + StandardScaler + PCA   (fit YALNIZCA train)
        ├─ PC1 (tek boyut)        → Otomata (beyaz kutu)
        └─ Çok değişkenli ölçekli → LSTM / GRU / 1D-CNN (kara kutu)
  └─ Değerlendirme: 3 senaryo × 5 tohum → metrikler + istatistik testleri
  └─ Açıklanabilirlik (otomata) + Görselleştirme + Rapor
```

- Merkezî `config.yaml` → kodda sabit sayı yok; SOLID/modüler pipeline.
- Tüm `fit` işlemleri yalnızca eğitimde → **veri sızıntısı yok.**

---

## Ön İşleme

- Eksik veri: SKAB dosya-içi, BATADAL **nedensel ileri-yön** (geleceğe sızma yok).
- `StandardScaler` + `PCA` yalnızca **eğitim** bölmesinde fit edilir.
- Otomata tek boyut ister → **PC1** (PCA birinci bileşeni).
- PC1 açıklanan varyans: SKAB ≈ %30,4, BATADAL ≈ %20,6.

> BATADAL'da tek bileşen düşük varyans taşır → otomata için bilgi kaybı (sonuçları etkiler).

---

## Beyaz Kutu — Olasılıksal Otomata

```
PC1 → z-norm → kayan pencere → PAA → SAX → durumlar
    → frekans tabanlı geçişler → Laplace yumuşatma → yol olasılığı
```

- Her benzersiz **SAX örüntüsü = bir durum (state)**.
- Geçiş olasılığı (Laplace, α=1):
  `P(Sᵢ→Sⱼ) = (sayım + α) / (toplam_çıkış + α·|durumlar|)`
- **Yol olasılığı** = ardışık geçişlerin çarpımı.
  Düşük olasılık → beklenmedik davranış → **yüksek anomali skoru.**

<!--
PAA pencereyi segment ortalamalarına indirger (bölme faktörü 2). SAX Gauss kesim
noktalarıyla sembole çevirir. z-norm global yapılır ki pencerenin mutlak seviyesi
korunsun (anomaliler çoğunlukla seviye sapması).
-->

---

## Görülmemiş Örüntü Yönetimi (Levenshtein)

- Eğitimden bir **SAX sözlüğü** çıkarılır.
- Test sırasında sözlükte olmayan örüntü → **unseen**.
- **Levenshtein** (düzenleme mesafesi) ile en yakın bilinen duruma eşlenir, mesafe oranında ceza eklenir.
- Bu mekanizma **zorunlu birim testlerle** doğrulanır (`test_levenshtein.py`, `test_unseen.py`).

> Kaydırılmış SKAB testinde 62 sözlük-dışı örüntü; en yakına ortalama mesafe **1,02** (en çok 2).

---

## Kara Kutu — Derin Öğrenme

Üç mimari, ortak girdi/çıktı sözleşmesi: girdi `(batch, 30, öznitelik)` → tek logit.

- **LSTM / GRU:** son zaman adımının gizli durumundan ikili sınıflandırma.
- **1D-CNN:** 2 evrişim + küresel maksimum havuzlama.
- Sabit eğitim: epoch ≤ 50, batch 32, lr 0,001, erken durdurma (sabır 5), gizli 48.
- Dengesizlik `pos_weight` ile telafi; eşik doğrulamada F1-maksimize.

---

## Açıklanabilirlik Modülü (otomatanın iç yapısından)

Her karar için üretilen **zorunlu JSON** (ister X.F):

```json
{ "time_step": 5, "state": "aab", "pattern": "adc",
  "status": "unseen", "mapped_to": "abc",
  "probability": 0.108, "decision": "anomaly" }
```

- Durum, örüntü, görüldü mü, eşleme, **geçişler + olasılıkları**, yol olasılığı, karar, **güven skoru.**
- Ek (bonus): **karşıt durum** (counterfactual) + **mesafe özeti** analizleri.
- Hem makine-okur JSON hem insan-okur Türkçe metin.

---

## Açıklanabilirlik — Gerçek Örnek (SKAB)

```
Durum (SAX) : cbcb
Yol         : bacb → acbc → cbcb
Skor / Eşik : 9.19 / 0.22  →  ANOMALİ (güven 1.00)
Açıklama    : Yol olasılığı 2.78e-04 (düşük → beklenmedik). 'acbc'
              örüntüsü görülmemiş; en yakın bilinen 'aabc'
              (Levenshtein=1), +1.0 ceza. Skor eşiği aştı → ANOMALİ.
```

> Derin öğrenme bu tür bir gerekçe **üretemez** — projenin temel katkısı budur.

---

## Deney Kurulumu

- **3 senaryo:** `orijinal`, `gurultu` (σ=0,3 Gauss), `unseen` (×1,8 kazanç kayması).
- **5 tohum** [42, 123, 2026, 7, 999]; otomata deterministik → kat başına 1 kez.
- **SKAB:** `StratifiedGroupKFold` (dosya sızıntısı yok), fold ort. ± std.
- **BATADAL:** zaman sıralı %60/%20/%20.
- **İstatistik:** Wilcoxon (kat bazlı F1) + McNemar (eşleşmiş kararlar).
- Her koşunun config'i `results/kullanilan_config.yaml`'a kaydedilir.

---

## Sonuç 1 — Doğruluk (F1, orijinal)

| Model | SKAB F1 | BATADAL F1 |
|-------|--------:|-----------:|
| Otomata | 0,521 (recall **0,97**) | **0,100** |
| LSTM | 0,875 | 0,040 |
| GRU | 0,874 | 0,047 |
| 1D-CNN | **0,882** | 0,000 |

![h:320](../results/figurler/fig_f1_karsilastirma.png)

<!--
SKAB'da DL açık ara önde. Otomata yüksek recall–düşük precision (her şeye yakın
anomali der). BATADAL'da tablo tersine döner: otomata en iyi, CNN tamamen çöker.
-->

---

## Sonuç 1 — Yorum

- **SKAB:** derin öğrenme açık ara önde (≈0,88 vs 0,52).
  Otomata = yüksek recall (0,97) / düşük precision (0,36).
- **BATADAL:** tablo tersine döner → **otomata en iyi (0,100)**, 1D-CNN = 0,000.
  - CNN `roc_auc ≈ 0,05`: skorlar **sistematik ters sıralanmış** (çoğunluğa çöküş).
  - Sıralamada (ROC-AUC) LSTM 0,787 / GRU 0,762 önde → iyi sıralar ama tek eşikle F1'e çeviremez.

> Az veri + yüksek dengesizlikte sade/yorumlanabilir model pratik bir alternatif.

---

## Sonuç 2 — Dayanıklılık (gürültü / unseen)

![h:330](../results/figurler/fig_senaryo_dayaniklilik.png)

- Otomata SKAB'da neredeyse **sabit**: 0,521 / 0,522 / 0,512.
- BATADAL'da otomata **unseen** senaryoda en iyi (0,122) → Levenshtein eşlemesi işe yarar.

---

## Sonuç 3 — Parametre Duyarlılığı

![h:330](../results/figurler/fig_parametre_duyarlilik.png)

- En iyi **alfabe = 3**; F1 ≈ 0,519–0,523 (büyük ölçüde duyarsız).
- Durum sayısı 18 → 1157, geçiş yoğunluğu 0,43 → 0,004 (matris seyrekleşir).

> Karmaşıklığı artırmak doğruluğu iyileştirmiyor → **sadelik lehine** bulgu.

---

## İstatistiksel Testler

- **Wilcoxon** (otomata vs LSTM/GRU/1D-CNN): p = 0,0625 (n=5; tutarlı yönlü farkın alt sınırı). DL tüm katlarda üstün.
- **McNemar (SKAB, otomata vs 1D-CNN):** istatistik = 3296,4, p ≈ 0 (yalnız otomata doğru: 415; yalnız CNN: 4400). Geçerli.
- **McNemar (BATADAL): yorum dışı** — tüm DL dejenere (F1≈0); ham doğruluk dejenere modeli kayırır → `yorum_disi` bayrağı.

> Dengesiz veride **accuracy yanıltıcıdır; asıl ölçüt F1/PR**.

| Otomata vs 1D-CNN (SKAB) | ROC / PR (en iyi DL) |
|---|---|
| ![h:230](../results/figurler/fig_karmasiklik_matrisi.png) | ![h:230](../results/figurler/fig_roc_pr.png) |

---

## Otomatanın İç Yapısı (görselleştirme)

| Geçiş olasılık ısı haritası | Durum geçiş diyagramı |
|---|---|
| ![h:330](../results/figurler/fig_gecis_matrisi.png) | ![h:330](../results/figurler/fig_durum_diyagrami.png) |

> Her durum ve geçiş gözlemlenebilir → kara kutu değil, **cam kutu.**

---

## Tartışma — Doğruluk vs Açıklanabilirlik

- **SKAB:** DL doğrulukta önde, ama gerekçe veremez.
- **BATADAL:** beyaz kutu doğrulukta **da** öne geçebiliyor (az veri rejimi).
- Otomata: her kararı denetlenebilir + **çok daha hızlı eğitim** (EK Tablo5).
- Düşük precision bir eşik *hatası değil*; F1-optimal eşiğin dürüst sonucu (3/5 kat "hepsi anomali").

> Amaç tek "en iyi" model değil; **hangi yaklaşım ne zaman** sorusunun niceliksel yanıtı.

---

## Sonuç ve Katkılar

- İki veri seti, beyaz kutu vs 3 kara kutu, 3 senaryo, 5 tohum, istatistik testleri — **eksiksiz ve sızıntısız** düzenek.
- **Açıklanabilirlik modülü:** zorunlu JSON + counterfactual + mesafe özeti (bonus dahil).
- Merkezî config, modüler pipeline, **92 birim test**, CI, akademik rapor + görseller.
- Tüm sayılar **gerçek deney çıktısı** (`results/`), `run_experiments` ile tekrar üretilebilir.

**Demo:** `python -m scripts.demo_explain --veri skab`

### Teşekkürler — Sorular?
