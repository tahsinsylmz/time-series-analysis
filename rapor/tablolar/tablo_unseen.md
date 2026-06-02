## Sozluk-disi Yonetimi - Detection Rate / Mapping Accuracy (VI.A)

| Veri | Kat | Toplam konum | Sozluk-disi | Detection Rate | Map.Acc (tam) | Map.Acc (mesafe<=1) |
| --- | --- | --- | --- | --- | --- | --- |
| SKAB | 1 | 4454 | 62 | 0.014 | 0.000 | 0.226 |
| SKAB | 2 | 4463 | 20 | 0.004 | 0.000 | 0.200 |
| SKAB | 3 | 4488 | 24 | 0.005 | 0.000 | 0.042 |
| SKAB | 4 | 4492 | 70 | 0.016 | 0.000 | 0.157 |
| SKAB | 5 | 4395 | 5 | 0.001 | 0.000 | 0.000 |
| BATADAL | 1 | 827 | 2 | 0.002 | 0.000 | 0.000 |
| **SKAB ort.** | - | 4458 | 36 | **0.008** | **0.000** | **0.125** |
| **BATADAL ort.** | - | 827 | 2 | **0.002** | **0.000** | **0.000** |

*VI.A sozluk-disi (out-of-vocabulary) yonetimi (yalniz otomata). Detection Rate = sozluk-disi pattern tasiyan test konumu orani; Mapping Accuracy = Levenshtein ile eslenen en yakin bilinen pattern'in, genlik kaydirmasindan ONCEKI gercek pattern'e dogrulugu (tam esitlik / mesafe<=1). Derin ogrenme modellerinde kavramsal karsiligi yoktur.*
