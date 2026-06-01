## Wilcoxon Isaretli Sira Testi (SKAB, kat-bazli F1)

| Karsilastirma | Veri | Istatistik | p |
| --- | --- | --- | --- |
| automata vs lstm | SKAB | 0.000 | 0.0625 |
| automata vs gru | SKAB | 0.000 | 0.0625 |
| automata vs cnn1d | SKAB | 0.000 | 0.0625 |

## McNemar Testi

| Veri | Karsilastirma | Istatistik | p | Yalniz A dogru | Yalniz B dogru |
| --- | --- | --- | --- | --- | --- |
| SKAB | automata vs cnn1d | 3296.419 | 0.00e+00 | 415 | 4400 |
| BATADAL | automata vs cnn1d | 25.289 | 4.93e-07 | 3 | 35 |
