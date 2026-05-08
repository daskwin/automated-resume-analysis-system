# Automated Resume Analysis System

Веб-сервис для автоматического анализа резюме и сопоставления кандидатов с вакансиями.

Проект позволяет загрузить резюме, извлечь из него текст, определить профессиональную область кандидата, предсказать IT-роль, извлечь ключевые сущности и оценить соответствие резюме добавленным вакансиям.

## Возможности

- загрузка резюме в форматах `PDF`, `DOCX`, `TXT`;
- извлечение текста из файла;
- классификация резюме по общей профессиональной области;
- определение наиболее вероятной IT-роли;
- извлечение контактов, навыков, компаний, образования и локаций;
- добавление и хранение вакансий;
- расчет процента соответствия резюме вакансии;
- веб-интерфейс для демонстрации работы сервиса.

## Стек технологий

- Python
- FastAPI
- PostgreSQL
- SQLAlchemy
- HTML / CSS / JavaScript
- Scikit-learn
- Sentence Transformers
- Transformers / PyTorch
- Yandex Object Storage S3

## Структура проекта

```text
app/
├── api/                 # API endpoints
├── db/                  # database connection and models
├── models/              # model directories, downloaded from S3
├── services/            # business logic and ML services
├── static/              # CSS and JavaScript
├── templates/           # HTML templates
└── main.py              # FastAPI application

data/
└── processed/           # processed datasets for experiments

notebooks/               # experiments and model training notebooks
scripts/                 # utility scripts
```

## Модели

Модели не хранятся напрямую в GitHub, чтобы не перегружать репозиторий большими файлами.

При запуске приложения модели автоматически скачиваются из S3-хранилища в папку:

```text
app/models/
```

Используются следующие модели:

```text
app/models/embedding_model/
app/models/it_role_model/
app/models/resume_bert_ner_model_chunked/
```

Для скачивания моделей используются переменные окружения:

```env
S3_MODEL_BUCKET=
S3_EMBEDDING_MODEL_KEY=
S3_IT_ROLE_MODEL_KEY=
S3_NER_MODEL_KEY=
S3_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_DEFAULT_REGION=ru-central1
```

## Локальный запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/daskwin/automated-resume-analysis-system.git
cd automated-resume-analysis-system
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Создать `.env`

```bash
cp .env.example .env
```

Заполнить в `.env` параметры подключения к базе данных и S3-хранилищу.

Пример:

```env
POSTGRES_DB=resume_analysis
POSTGRES_USER=resume_analysis
POSTGRES_PASSWORD=resume_analysis_password
POSTGRES_PORT=5433

DATABASE_URL=postgresql+psycopg2://resume_analysis:resume_analysis_password@localhost:5433/resume_analysis

AWS_DEFAULT_REGION=ru-central1
S3_ENDPOINT_URL=https://storage.yandexcloud.net

S3_MODEL_BUCKET=your_bucket_name
S3_EMBEDDING_MODEL_KEY=VKR/embedding_model.zip
S3_IT_ROLE_MODEL_KEY=VKR/it_role_model.zip
S3_NER_MODEL_KEY=VKR/resume_bert_ner_model_chunked.zip
```

### 5. Запустить PostgreSQL

```bash
docker compose -f docker-compose.postgres.yml up -d
```

### 6. Запустить приложение

```bash
python -m uvicorn app.main:app --reload
```

После запуска приложение будет доступно по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```
