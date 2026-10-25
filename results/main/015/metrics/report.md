# Отчёт по метрикам

- Запуск: `015`
- Выборка: `main`
- Рассчитано (UTC): `2026-10-25T09:47:02.103351+00:00`
- Версия алгоритма: `1.2.0`
- Выбрано поединков: 25

## Основные результаты

| Модель | Скоуп | Accuracy | 95% bootstrap CI | Cohen's κ | Macro-F1 | N поединков |
|---|---|---:|---:|---:|---:|---:|
| qwen3-8-flash | model:qwen3-8-flash | 66.67% | 56.00%–77.33% | 0.3194 | 0.6546 | 25 |
| qwen3-8-flash | common:qwen3-8-flash | 66.67% | 56.00%–77.33% | 0.3194 | 0.6546 | 25 |
| clef | model:clef | 68.00% | 56.00%–78.67% | 0.3455 | 0.6667 | 25 |
| clef | common:clef | 68.00% | 56.00%–78.67% | 0.3455 | 0.6667 | 25 |

## Сравнение моделей

`clef − qwen3-8-flash` на общей выборке: 1.33 п.п.; 95% CI: -10.67–13.33 п.п..

## Проверка гипотез H1–H3

Решения основаны только на семи заранее заданных перестановочных тестах с коррекцией Holm при α = 0,05. Bootstrap CI являются описательными.

| Гипотеза | Тест | Эффект | 95% bootstrap CI | raw p | Holm p | N поединков | Решение |
|---|---|---:|---:|---:|---:|---:|---|
| H1 | `H1:qwen3-8-flash:agreement_rate` | 0.6667 | 0.5600…0.7733 | 0.005690 | 0.031500 | 25 | confirmed |
| H1 | `H1:qwen3-8-flash:cohens_kappa` | 0.3194 | 0.0675…0.5311 | 0.005590 | 0.031500 | 25 | confirmed |
| H1 | `H1:clef:agreement_rate` | 0.6800 | 0.5600…0.7867 | 0.004530 | 0.031500 | 25 | confirmed |
| H1 | `H1:clef:cohens_kappa` | 0.3455 | 0.1000…0.5562 | 0.004500 | 0.031500 | 25 | confirmed |
| H2 | `H2:accuracy_difference` | 1.3333 | -10.6667…13.3333 | 1.000000 | 1.000000 | 25 | not_confirmed |
| H3 | `H3:pairwise_repeatability_difference` | 10.6667 | 4.4444…18.6667 | 0.007470 | 0.031500 | 25 | confirmed |
| H3 | `H3:label_flip_rate_difference` | 8.0000 | -6.6667…24.0000 | 0.419456 | 0.838912 | 25 | confirmed |

Итоговые решения:

- **H1:qwen3-8-flash: confirmed.** Accuracy выше 0,5 и Cohen's kappa выше 0 после коррекции Holm.
- **H1:clef: confirmed.** Accuracy выше 0,5 и Cohen's kappa выше 0 после коррекции Holm.
- **H2: not_confirmed.** Статистически значимая разница Accuracy моделей не установлена.
- **H3: confirmed.** Хотя бы одно различие устойчивости значимо после коррекции Holm.

## Согласованность людей

| Скоуп | Pairwise agreement | Fleiss' κ | N поединков |
|---|---:|---:|---:|
| all_humans | 59.11% | 0.1803 | 25 |
| humans:model:qwen3-8-flash | 59.11% | 0.1803 | 25 |
| humans:model:clef | 59.11% | 0.1803 | 25 |
| humans:common | 59.11% | 0.1803 | 25 |

## Согласованность с отдельными судьями

| Модель | Скоуп | Majority-to-individual | All-pairs |
|---|---|---:|---:|
| qwen3-8-flash | model:qwen3-8-flash | 60.44% | 61.04% |
| qwen3-8-flash | common:qwen3-8-flash | 60.44% | 61.04% |
| clef | model:clef | 60.00% | 60.00% |
| clef | common:clef | 60.00% | 60.00% |

## Разбивка по коллегиям

Подробные значения Agreement Rate и позиционных метрик находятся в `by_collegium.csv`.

## Устойчивость

| Модель | Скоуп | Единогласие | Попарная повторяемость |
|---|---|---:|---:|
| qwen3-8-flash | model:qwen3-8-flash | 84.00% | 89.33% |
| qwen3-8-flash | common:qwen3-8-flash | 84.00% | 89.33% |
| clef | model:clef | 100.00% | 100.00% |
| clef | common:clef | 100.00% | 100.00% |

## Позиционная чувствительность

Интерпретировать эти показатели следует совместно с повторяемостью модели.

| Модель | Скоуп | Label Flip Rate | Vote Share Change | N поединков |
|---|---|---:|---:|---:|
| qwen3-8-flash | position:qwen3-8-flash | 16.00% | 18.67% | 25 |
| qwen3-8-flash | position:common:qwen3-8-flash | 16.00% | 18.67% | 25 |
| clef | position:clef | 24.00% | 24.44% | 25 |
| clef | position:common:clef | 24.00% | 24.44% | 25 |

## Вероятностные метрики

| Модель | Скоуп | Coverage | Brier | Brier individual | Vote Share MAE | MAE individual |
|---|---|---:|---:|---:|---:|---:|
| qwen3-8-flash | model:qwen3-8-flash | 0.00% | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) |
| qwen3-8-flash | common:qwen3-8-flash | 0.00% | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) |
| clef | model:clef | 100.00% | 0.2295 | 0.2295 | 0.2611 | 0.2611 |
| clef | common:clef | 100.00% | 0.2295 | 0.2295 | 0.2611 | 0.2611 |

## Описательные итоги по всем голосам

| Модель | Полных поединков | Совпадений общего победителя | Доля |
|---|---:|---:|---:|
| qwen3-8-flash | 25 | 17 | 68.00% |
| clef | 25 | 16 | 64.00% |

| Модель | Поединок | Счёт людей (P1:P2) | Победитель людей | Счёт модели (P1:P2) | Победитель модели | Совпадение |
|---|---|---:|---|---:|---|---:|
| qwen3-8-flash | 6 | 5:4 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 7 | 2:7 | player_2 | 0:9 | player_2 | True |
| qwen3-8-flash | 8 | 5:4 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 9 | 8:1 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 10 | 3:6 | player_2 | 5:4 | player_1 | False |
| qwen3-8-flash | 11 | 8:1 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 12 | 1:8 | player_2 | 0:9 | player_2 | True |
| qwen3-8-flash | 13 | 1:8 | player_2 | 4:5 | player_2 | True |
| qwen3-8-flash | 14 | 6:3 | player_1 | 6:3 | player_1 | True |
| qwen3-8-flash | 15 | 6:3 | player_1 | 2:7 | player_2 | False |
| qwen3-8-flash | 16 | 1:8 | player_2 | 0:9 | player_2 | True |
| qwen3-8-flash | 17 | 7:2 | player_1 | 3:6 | player_2 | False |
| qwen3-8-flash | 18 | 6:3 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 19 | 2:7 | player_2 | 9:0 | player_1 | False |
| qwen3-8-flash | 20 | 3:6 | player_2 | 6:3 | player_1 | False |
| qwen3-8-flash | 21 | 7:2 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 22 | 5:4 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 23 | 3:6 | player_2 | 9:0 | player_1 | False |
| qwen3-8-flash | 24 | 9:0 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 25 | 1:8 | player_2 | 2:7 | player_2 | True |
| qwen3-8-flash | 26 | 5:4 | player_1 | 8:1 | player_1 | True |
| qwen3-8-flash | 27 | 6:3 | player_1 | 7:2 | player_1 | True |
| qwen3-8-flash | 28 | 5:4 | player_1 | 4:5 | player_2 | False |
| qwen3-8-flash | 29 | 5:4 | player_1 | 9:0 | player_1 | True |
| qwen3-8-flash | 30 | 8:1 | player_1 | 1:8 | player_2 | False |
| clef | 6 | 5:4 | player_1 | 9:0 | player_1 | True |
| clef | 7 | 2:7 | player_2 | 0:9 | player_2 | True |
| clef | 8 | 5:4 | player_1 | 6:3 | player_1 | True |
| clef | 9 | 8:1 | player_1 | 9:0 | player_1 | True |
| clef | 10 | 3:6 | player_2 | 6:3 | player_1 | False |
| clef | 11 | 8:1 | player_1 | 9:0 | player_1 | True |
| clef | 12 | 1:8 | player_2 | 0:9 | player_2 | True |
| clef | 13 | 1:8 | player_2 | 6:3 | player_1 | False |
| clef | 14 | 6:3 | player_1 | 9:0 | player_1 | True |
| clef | 15 | 6:3 | player_1 | 0:9 | player_2 | False |
| clef | 16 | 1:8 | player_2 | 0:9 | player_2 | True |
| clef | 17 | 7:2 | player_1 | 3:6 | player_2 | False |
| clef | 18 | 6:3 | player_1 | 9:0 | player_1 | True |
| clef | 19 | 2:7 | player_2 | 6:3 | player_1 | False |
| clef | 20 | 3:6 | player_2 | 9:0 | player_1 | False |
| clef | 21 | 7:2 | player_1 | 9:0 | player_1 | True |
| clef | 22 | 5:4 | player_1 | 9:0 | player_1 | True |
| clef | 23 | 3:6 | player_2 | 9:0 | player_1 | False |
| clef | 24 | 9:0 | player_1 | 9:0 | player_1 | True |
| clef | 25 | 1:8 | player_2 | 3:6 | player_2 | True |
| clef | 26 | 5:4 | player_1 | 9:0 | player_1 | True |
| clef | 27 | 6:3 | player_1 | 9:0 | player_1 | True |
| clef | 28 | 5:4 | player_1 | 0:9 | player_2 | False |
| clef | 29 | 5:4 | player_1 | 3:6 | player_2 | False |
| clef | 30 | 8:1 | player_1 | 9:0 | player_1 | True |

## Исключения и N/A

Зафиксировано исключений: 150. Полный перечень — в `exclusions.csv`; диагностика кейсов — в `per_bout_collegium.csv`.
- No eligible bout-collegium cases contain every planned provider probability

## Ограничения интерпретации

- Решения людей используются как референс для согласованности, но не объявляются объективной истиной.
- SHA-256 текущих транскрипций не доказывают, что в момент оценивания использовался идентичный текст.
- Position bias необходимо интерпретировать вместе с repeatability: смена решения может отражать общую нестабильность.
- Bootstrap-интервалы не устраняют возможную нерепрезентативность выборки.

Подробные машинно-читаемые результаты и знаменатели находятся в `metrics.json` и CSV-файлах.
