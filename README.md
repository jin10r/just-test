# Telegram Rental Parser (Pyrogram + ruBERT + FastAPI)

Парсер постов телеграм-каналов на русском языке. Извлекает:
- до 3 фото
- станция метро
- адрес (с геокодированием через Nominatim)
- цена
- телефон
- количество комнат
- этаж
- остальное описание

Технологии: Pyrogram (готовая user session), ruBERT tiny NER, Natasha, FastAPI. БД: MongoDB (по env), при её отсутствии можно выгружать в CSV.

## Переменные окружения
- MONGO_URL — строка подключения MongoDB (если нет — запись в БД пропускается)
- MONGO_DB_NAME — имя базы (по умолчанию tg_parser)
- PYROGRAM_SESSION_DIR — папка с сессией (по умолчанию /app/sessions)
- PYROGRAM_SESSION_NAME — имя файла сессии без расширения (по умолчанию user)
- IMAGES_DIR — путь для сохранения изображений (по умолчанию /app/data/images)
- EXPORTS_DIR — путь для CSV (по умолчанию /app/data/exports)
- NOMINATIM_EMAIL — опционально, попадёт в User-Agent для Nominatim
- TG_API_ID / TG_API_HASH — опционально, если Pyrogram в окружении требует app credentials

## API (префикс /api)
- GET /api/health
- POST /api/parse {"channel": "arendakv_msk", "limit": 30}
  - возвращает {inserted, updated, count, errors}, элементы не возвращает (чтобы не грузить ответ)
- POST /api/parse_to_csv {"channel": "arendakv_msk", "limit": 30, "csv_name": "optional.csv"}
  - парсит, пишет CSV в EXPORTS_DIR и возвращает путь
- GET /api/posts — читает из БД, если MONGO_URL задан

## CLI
```
python3 scripts/parse_cli.py https://t.me/arendakv_msk --limit 30
```

## Сессия Pyrogram
Положите готовый user session файл в `${PYROGRAM_SESSION_DIR}/${PYROGRAM_SESSION_NAME}.session`.

## CSV формат
Столбцы: channel, message_id, date, photos (до 3 путей, разделены "; "), metro, address, price, phone, rooms, floor, description.