# Отчёт по метрикам

> **МЕТОДОЛОГИЧЕСКИ НЕПРИГОДЕН ДЛЯ ИТОГОВЫХ ВЫВОДОВ.** В запуске `011`
> адаптеры до вызова модели приводили `original` и `swapped` к одному
> каноническому порядку игроков. Это искусственно устраняло измеряемый
> позиционный эффект и меняло заданный протокол. Запуск сохранён только как
> технический журнал и не должен использоваться для проверки гипотез H1–H3.

- Запуск: `011`
- Выборка: `main`
- Рассчитано (UTC): `2026-10-24T11:38:11.008154+00:00`
- Версия алгоритма: `1.2.0`
- Выбрано поединков: 25

## Основные результаты

| Модель | Скоуп | Accuracy | 95% bootstrap CI | Cohen's κ | Macro-F1 | N поединков |
|---|---|---:|---:|---:|---:|---:|
| qwen3-7-flash | model:qwen3-7-flash | 62.67% | 49.33%–74.67% | 0.2419 | 0.6185 | 25 |
| qwen3-7-flash | common:qwen3-7-flash | 62.67% | 49.33%–74.67% | 0.2419 | 0.6185 | 25 |
| clef | model:clef | 64.00% | 53.33%–74.67% | 0.2676 | 0.6305 | 25 |
| clef | common:clef | 64.00% | 53.33%–74.67% | 0.2676 | 0.6305 | 25 |

## Сравнение моделей

`clef − qwen3-7-flash` на общей выборке: 1.33 п.п.; 95% CI: -8.00–13.33 п.п..

## Проверка гипотез H1–H3

Решения основаны только на семи заранее заданных перестановочных тестах с коррекцией Holm при α = 0,05. Bootstrap CI являются описательными.

| Гипотеза | Тест | Эффект | 95% bootstrap CI | raw p | Holm p | N поединков | Решение |
|---|---|---:|---:|---:|---:|---:|---|
| H1 | `H1:qwen3-7-flash:agreement_rate` | 0.6267 | 0.4933…0.7467 | 0.039450 | 0.191998 | 25 | not_confirmed |
| H1 | `H1:qwen3-7-flash:cohens_kappa` | 0.2419 | -0.0241…0.4824 | 0.038400 | 0.191998 | 25 | not_confirmed |
| H1 | `H1:clef:agreement_rate` | 0.6400 | 0.5333…0.7467 | 0.015590 | 0.104929 | 25 | not_confirmed |
| H1 | `H1:clef:cohens_kappa` | 0.2676 | 0.0482…0.4721 | 0.014990 | 0.104929 | 25 | not_confirmed |
| H2 | `H2:accuracy_difference` | 1.3333 | -8.0000…13.3333 | 1.000000 | 1.000000 | 25 | not_confirmed |
| H3 | `H3:pairwise_repeatability_difference` | 7.1111 | -1.7778…16.0000 | 0.195048 | 0.585144 | 25 | not_confirmed |
| H3 | `H3:label_flip_rate_difference` | -4.0000 | -10.6667…0.0000 | 0.503065 | 1.000000 | 25 | not_confirmed |

Итоговые решения:

- **H1:qwen3-7-flash: not_confirmed.** Совместный критерий Accuracy и Cohen's kappa не выполнен.
- **H1:clef: not_confirmed.** Совместный критерий Accuracy и Cohen's kappa не выполнен.
- **H2: not_confirmed.** Статистически значимая разница Accuracy моделей не установлена.
- **H3: not_confirmed.** Значимые различия устойчивости моделей не установлены.

## Согласованность людей

| Скоуп | Pairwise agreement | Fleiss' κ | N поединков |
|---|---:|---:|---:|
| all_humans | 59.11% | 0.1803 | 25 |
| humans:model:qwen3-7-flash | 59.11% | 0.1803 | 25 |
| humans:model:clef | 59.11% | 0.1803 | 25 |
| humans:common | 59.11% | 0.1803 | 25 |

## Согласованность с отдельными судьями

| Модель | Скоуп | Majority-to-individual | All-pairs |
|---|---|---:|---:|
| qwen3-7-flash | model:qwen3-7-flash | 56.44% | 56.59% |
| qwen3-7-flash | common:qwen3-7-flash | 56.44% | 56.59% |
| clef | model:clef | 59.56% | 59.41% |
| clef | common:clef | 59.56% | 59.41% |

## Разбивка по коллегиям

Подробные значения Agreement Rate и позиционных метрик находятся в `by_collegium.csv`.

## Устойчивость

| Модель | Скоуп | Единогласие | Попарная повторяемость |
|---|---|---:|---:|
| qwen3-7-flash | model:qwen3-7-flash | 85.33% | 90.22% |
| qwen3-7-flash | common:qwen3-7-flash | 85.33% | 90.22% |
| clef | model:clef | 96.00% | 97.33% |
| clef | common:clef | 96.00% | 97.33% |

## Позиционная чувствительность

Интерпретировать эти показатели следует совместно с повторяемостью модели.

| Модель | Скоуп | Label Flip Rate | Vote Share Change | N поединков |
|---|---|---:|---:|---:|
| qwen3-7-flash | position:qwen3-7-flash | 4.00% | 4.89% | 25 |
| qwen3-7-flash | position:common:qwen3-7-flash | 4.00% | 4.89% | 25 |
| clef | position:clef | 0.00% | 0.89% | 25 |
| clef | position:common:clef | 0.00% | 0.89% | 25 |

## Вероятностные метрики

| Модель | Скоуп | Coverage | Brier | Brier individual | Vote Share MAE | MAE individual |
|---|---|---:|---:|---:|---:|---:|
| qwen3-7-flash | model:qwen3-7-flash | 0.00% | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) |
| qwen3-7-flash | common:qwen3-7-flash | 0.00% | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) | N/A (No eligible bout-collegium cases contain every planned provider probability) |
| clef | model:clef | 100.00% | 0.2437 | 0.2441 | 0.2744 | 0.2748 |
| clef | common:clef | 100.00% | 0.2437 | 0.2441 | 0.2744 | 0.2748 |

## Описательные итоги по всем голосам

| Модель | Полных поединков | Совпадений общего победителя | Доля |
|---|---:|---:|---:|
| qwen3-7-flash | 25 | 15 | 60.00% |
| clef | 25 | 17 | 68.00% |

| Модель | Поединок | Счёт людей (P1:P2) | Победитель людей | Счёт модели (P1:P2) | Победитель модели | Совпадение |
|---|---|---:|---|---:|---|---:|
| qwen3-7-flash | 6 | 5:4 | player_1 | 4:5 | player_2 | False |
| qwen3-7-flash | 7 | 2:7 | player_2 | 1:8 | player_2 | True |
| qwen3-7-flash | 8 | 5:4 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 9 | 8:1 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 10 | 3:6 | player_2 | 9:0 | player_1 | False |
| qwen3-7-flash | 11 | 8:1 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 12 | 1:8 | player_2 | 0:9 | player_2 | True |
| qwen3-7-flash | 13 | 1:8 | player_2 | 6:3 | player_1 | False |
| qwen3-7-flash | 14 | 6:3 | player_1 | 6:3 | player_1 | True |
| qwen3-7-flash | 15 | 6:3 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 16 | 1:8 | player_2 | 0:9 | player_2 | True |
| qwen3-7-flash | 17 | 7:2 | player_1 | 0:9 | player_2 | False |
| qwen3-7-flash | 18 | 6:3 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 19 | 2:7 | player_2 | 9:0 | player_1 | False |
| qwen3-7-flash | 20 | 3:6 | player_2 | 0:9 | player_2 | True |
| qwen3-7-flash | 21 | 7:2 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 22 | 5:4 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 23 | 3:6 | player_2 | 9:0 | player_1 | False |
| qwen3-7-flash | 24 | 9:0 | player_1 | 9:0 | player_1 | True |
| qwen3-7-flash | 25 | 1:8 | player_2 | 6:3 | player_1 | False |
| qwen3-7-flash | 26 | 5:4 | player_1 | 6:3 | player_1 | True |
| qwen3-7-flash | 27 | 6:3 | player_1 | 3:6 | player_2 | False |
| qwen3-7-flash | 28 | 5:4 | player_1 | 2:7 | player_2 | False |
| qwen3-7-flash | 29 | 5:4 | player_1 | 8:1 | player_1 | True |
| qwen3-7-flash | 30 | 8:1 | player_1 | 0:9 | player_2 | False |
| clef | 6 | 5:4 | player_1 | 0:9 | player_2 | False |
| clef | 7 | 2:7 | player_2 | 3:6 | player_2 | True |
| clef | 8 | 5:4 | player_1 | 2:7 | player_2 | False |
| clef | 9 | 8:1 | player_1 | 9:0 | player_1 | True |
| clef | 10 | 3:6 | player_2 | 9:0 | player_1 | False |
| clef | 11 | 8:1 | player_1 | 9:0 | player_1 | True |
| clef | 12 | 1:8 | player_2 | 0:9 | player_2 | True |
| clef | 13 | 1:8 | player_2 | 6:3 | player_1 | False |
| clef | 14 | 6:3 | player_1 | 6:3 | player_1 | True |
| clef | 15 | 6:3 | player_1 | 0:9 | player_2 | False |
| clef | 16 | 1:8 | player_2 | 0:9 | player_2 | True |
| clef | 17 | 7:2 | player_1 | 3:6 | player_2 | False |
| clef | 18 | 6:3 | player_1 | 9:0 | player_1 | True |
| clef | 19 | 2:7 | player_2 | 9:0 | player_1 | False |
| clef | 20 | 3:6 | player_2 | 0:9 | player_2 | True |
| clef | 21 | 7:2 | player_1 | 9:0 | player_1 | True |
| clef | 22 | 5:4 | player_1 | 9:0 | player_1 | True |
| clef | 23 | 3:6 | player_2 | 9:0 | player_1 | False |
| clef | 24 | 9:0 | player_1 | 9:0 | player_1 | True |
| clef | 25 | 1:8 | player_2 | 3:6 | player_2 | True |
| clef | 26 | 5:4 | player_1 | 9:0 | player_1 | True |
| clef | 27 | 6:3 | player_1 | 7:2 | player_1 | True |
| clef | 28 | 5:4 | player_1 | 6:3 | player_1 | True |
| clef | 29 | 5:4 | player_1 | 9:0 | player_1 | True |
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
