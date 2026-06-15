# Semantic Search Reranking for Cybersecurity

Репозиторий содержит реализацию двухэтапного алгоритма семантического поиска в области кибербезопасности на основе bi-encoder и cross-encoder моделей. Проект был разработан в рамках дипломной работы при исследовании методов повышения точности поиска с использованием моделей реранжирования.

---

## Архитектура алгоритма

Поиск выполняется в два этапа:

1. Запрос и документы кодируются bi-encoder моделью.
2. Векторный поиск выполняется в индексе FAISS.
3. Из базы извлекаются Top-N наиболее похожих документов.
4. Для каждой пары «запрос–документ» вычисляется оценка релевантности cross-encoder моделью.
5. Документы переупорядочиваются по оценкам релевантности.
6. Пользователю возвращаются Top-K результатов после реранжирования.

Схема позволяет объединить высокую скорость поиска bi-encoder моделей и высокую точность cross-encoder моделей.

---

## Используемые модели

### Bi-encoder

Основная модель поиска:

- `sentence-transformers/all-MiniLM-L6-v2`

Дополнительно в экспериментах сравнивались:

- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/e5-base-v2`
- `intfloat/e5-large-v2`
- `BAAI/bge-base-en-v1.5`

### Cross-encoder

Базовая модель:

- `cross-encoder/ms-marco-MiniLM-L6-v2`

Итоговая модель реранжирования, показавшая лучшие результаты:

- [jobby32/ms-marco-cybersecurity-MiniLM-L6-v2](https://huggingface.co/jobby32/ms-marco-cybersecurity-MiniLM-L6-v2)

Данная модель была дообучена на предметной области кибербезопасности и опубликована на Hugging Face.

---

## Датасет

Для обучения, валидации и тестирования использовался датасет:

- `jobby32/cybersecurity-QA-with-negatives`

Итоговая экспериментальная оценка качества поиска выполнялась на отложенной тестовой выборке:

- `final_test_holdout.csv`

Файл `final_test_holdout.csv` формируется из датасета `jobby32/cybersecurity-QA-with-negatives` и не используется при обучении моделей.

---

## Структура проекта

### `semantic_search_rerank.py`

Реализация двухэтапного поиска:

- построение FAISS индекса;
- поиск кандидатов bi-encoder моделью;
- реранжирование результатов cross-encoder моделью;
- вывод итоговой выдачи.

Пример запуска:

```bash
python semantic_search_rerank.py \
    --documents_csv final_test_holdout.csv \
    --query "What is SQL injection?"
```

---

### `comparison_pipelines.py`

Экспериментальный модуль для сравнения различных конфигураций поиска.

Поддерживает:

- сравнение нескольких bi-encoder моделей;
- сравнение нескольких моделей реранжирования;
- вычисление MRR и HitRate;
- измерение времени поиска и реранжирования;
- сохранение результатов в CSV.

Пример запуска:

```bash
python comparison_pipelines.py
```

---

### `train_cross_encoder_ranknet.py`

Дообучение cross-encoder модели с использованием функции потерь RankNet.

Возможности:

- обучение на парах positive / hard negative;
- ранняя остановка (Early Stopping);
- сохранение лучшей модели;
- построение графиков обучения;
- сохранение конфигурации эксперимента.

Пример запуска:

```bash
python train_cross_encoder_ranknet.py \
    --train_csv train.csv \
    --val_csv validation.csv
```

---

### `train_cross_encoder.py`

Дообучение cross-encoder модели с использованием функции потерь OrderedTripletLoss.

Возможности:

- обучение на триплетах;
- автоматический Grid Search параметров функции потерь;
- выбор лучшей конфигурации;
- сохранение истории обучения;
- сохранение лучшей модели.

Пример запуска:

```bash
python train_cross_encoder.py \
    --train_csv train.csv \
    --val_csv validation.csv
```

---

## Установка

Создание окружения:

```bash
python -m venv venv
source venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

---

## Метрики

Для оценки качества поиска используются:

- MRR@15 (Mean Reciprocal Rank)
- HitRate@3
- HitRate@5
- HitRate@10

---

## Основной результат

Проведённые эксперименты показали, что использование двухэтапного поиска с реранжированием позволяет существенно повысить качество поиска по сравнению с использованием только bi-encoder моделей.

Наилучшие результаты были получены для конфигурации:

- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder: `jobby32/ms-marco-cybersecurity-MiniLM-L6-v2`

Данная конфигурация использовалась в качестве итоговой версии алгоритма семантического поиска в области кибербезопасности.