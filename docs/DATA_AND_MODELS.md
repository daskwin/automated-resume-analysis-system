# Данные и обучение моделей

В проекте используются несколько независимых пайплайнов подготовки данных и обучения моделей.

## Общая структура

```text
data/
├── raw/                 # исходные датасеты
└── processed/           # обработанные датасеты

scripts/                 # скрипты подготовки датасетов
notebooks/               # ноутбуки обучения моделей

app/models/              # директория для моделей
├── embedding_model/
├── it_role_model/
└── resume_bert_ner_model_chunked/
```

Модельные артефакты хранятся в Yandex Object Storage S3 и автоматически скачиваются при запуске.

## Используемые модели

В сервисе используются три основные модели:

| Модель | Назначение | Директория |
|---|---|---|
| `embedding_model` | определение общей профессиональной области кандидата | `app/models/embedding_model/` |
| `it_role_model` | определение конкретной IT-роли кандидата | `app/models/it_role_model/` |
| `resume_bert_ner_model_chunked` | извлечение сущностей из резюме | `app/models/resume_bert_ner_model_chunked/` |

## Подготовка датасета для общей классификации резюме

Для общей классификации резюме используется датасет `Darshan-04/Resume-classification` с [Hugging Face](https://huggingface.co/datasets/Darshan-04/Resume-classification).

Исходный датасет содержит тексты резюме и профессиональные категории. На первом этапе данные приводятся к единому формату проекта:

```text
resume_text     # текст резюме
target_role     # исходная категория
source          # источник данных
```

Подготовка выполняется скриптом:

```bash
python scripts/download_resume_dataset.py
```

На выходе формируются файлы:

```text
data/raw/resume_dataset_hf_raw.csv
data/processed/resume_dataset.csv
```

Файл `resume_dataset.csv` используется как базовая обработанная версия исходного датасета.

## Группировка категорий для общей модели

Изначальные категории резюме достаточно детальные: `ACCOUNTANT`, `FINANCE`, `BANKING`, `ENGINEERING`, `INFORMATION-TECHNOLOGY`, `SALES`, `HR` и др.

Для финальной версии сервиса эти категории были объединены в более крупные профессиональные направления:

```text
BUSINESS
CREATIVE
FINANCE
LEGAL
OPERATIONS
PEOPLE
SERVICE
TECHNICAL
```

Группировка выполняется скриптом:

```bash
python scripts/prepare_general_grouped_dataset.py
```

На выходе формируется файл:

```text
data/processed/resume_dataset_general_grouped.csv
```

Именно этот файл используется для обучения финальной модели общей классификации.

## Baseline-модель TF-IDF

В качестве начальной точки для сравнения используется baseline-модель:

```text
TF-IDF + LogisticRegression
```

Baseline обучается в ноутбуке:

```text
notebooks/00_train_tfidf_baseline.ipynb
```

Эта модель не является основной моделью сервиса. Она используется как начальная точка для сравнения классического подхода к текстовой классификации с embedding-based моделями.

Baseline обучается на:

```text
data/processed/resume_dataset_general_grouped.csv
```

## Обучение модели общей классификации

Обучение общей модели выполняется в ноутбуке:

```text
notebooks/01_train_resume_category_classifier.ipynb
```

Используемый датасет:

```text
data/processed/resume_dataset_general_grouped.csv
```

В ноутбуке сравниваются разные подходы:

- TF-IDF baseline;
- sentence embeddings;
- LogisticRegression;
- LinearSVC;
- подбор гиперпараметров;
- дополнительные варианты с chunked embeddings, hybrid и ensemble-подходами.

Финальная модель:

```text
BAAI/bge-small-en-v1.5 + LinearSVC
```

Параметры финального классификатора:

```text
C = 3
class_weight = None
loss = squared_hinge
```

Итоговые классы модели:

```text
BUSINESS
CREATIVE
FINANCE
LEGAL
OPERATIONS
PEOPLE
SERVICE
TECHNICAL
```

Артефакты сохраняются в директорию:

```text
app/models/embedding_model/
```

Ожидаемые файлы:

```text
embedding_classifier.pkl
label_encoder.pkl
metadata.pkl
```

## Старые эксперименты с несгруппированными категориями

В директории `notebooks/` также есть ноутбуки с суффиксом `_old`.

Эти ноутбуки относятся к ранним экспериментам, где модель общей классификации обучалась на исходных несгруппированных категориях датасета `Darshan-04/Resume-classification`.

Эти эксперименты были сохранены для истории и сравнения, но не используются как финальная версия пайплайна сервиса.

## Подготовка датасета для классификации IT-ролей

Для определения конкретной IT-роли используется несколько источников данных:

1. [Resume Dataset на Kaggle](https://www.kaggle.com/datasets/avishekmajhi/resume-dataset);

2. [UpdatedResumeDataSet на Kaggle](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset);

3. синтетические шаблонные примеры для редких IT-ролей.


Перед запуском скрипта исходные CSV-файлы необходимо скачать вручную и положить в директорию `data/raw/`.

Ожидаемая структура входных файлов:

```text

data/raw/

├── kaggle_avishekmajhi/

│   └── Resume.csv

└── kaggle_updated_resume/

    └── UpdatedResumeDataSet.csv
    
    
Подготовка выполняется скриптом:

```bash
python scripts/prepare_it_roles_dataset.py
```

На выходе формируется файл:

```text
data/processed/resume_dataset_it_roles.csv
```

Итоговые классы включают, например:

```text
Data Engineer
Data Scientist
Python Developer
Java Developer
Frontend Developer
DevOps Engineer
QA Engineer
Security Engineer
Database Administrator
Project Manager IT
Business Analyst IT
```

## Обучение модели IT-роли

Обучение модели IT-роли выполняется в ноутбуке:

```text
notebooks/02_train_it_role_classifier.ipynb
```

Используемый датасет:

```text
data/processed/resume_dataset_it_roles.csv
```

В ноутбуке строятся embeddings с помощью:

```text
BAAI/bge-small-en-v1.5
```

Затем сравниваются классификаторы:

- `LogisticRegression`;
- `LinearSVC`.

Для моделей выполняется подбор гиперпараметров по метрике `macro_f1`.

`LinearSVC` рассматривался как лучший вариант по метрике macro_f1, но в финальный сервис была включена `LogisticRegression` как более удобная и интерпретируемая модель для прикладного использования.

## Обучение NER-модели

NER-модель обучается на специализированном датасете [Resume NER Training Dataset](https://www.kaggle.com/datasets/yashpwrr/resume-ner-training-dataset).

Перед запуском ноутбука исходный датасет необходимо скачать вручную с Kaggle и разместить в директории data/raw/.

Обучение выполняется в ноутбуке:

```text
notebooks/03_train_resume_ner_model.ipynb
```

Финальные артефакты сохраняются в директорию:

```text
app/models/resume_bert_ner_model_chunked/
```

Ожидаемые файлы включают:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
metadata.json
```

## Полный порядок воспроизведения обучения

Примерный порядок воспроизведения пайплайна:

```bash
# 1. Подготовить базовый датасет резюме
python scripts/prepare_resume_dataset.py

# 2. Подготовить grouped-датасет для общей классификации
python scripts/prepare_general_grouped_dataset.py

# 3. Подготовить датасет IT-ролей
python scripts/prepare_it_roles_dataset.py

# 4. Обучить baseline
jupyter notebook notebooks/00_train_tfidf_baseline.ipynb

# 5. Обучить модель общей классификации
jupyter notebook notebooks/01_train_resume_category_classifier.ipynb

# 6. Обучить модель IT-роли
jupyter notebook notebooks/02_train_it_role_classifier.ipynb

# 7. Обучить NER-модель
jupyter notebook notebooks/03_train_resume_ner_model.ipynb
```