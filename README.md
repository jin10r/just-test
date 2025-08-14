# Telegram Rental Parser (Watcher + Dynamic Schema)

Функционал:
- Парсинг телеграм-каналов через Pyrogram (user session)
- Извлечение параметров из неструктурированного caption: рукописные правила + BERT QA
- Геокодирование через Nominatim
- До 3 фото с поста
- Хранение в MongoDB (UUID как id), CSV экспорт по запросу
- Мониторинг списка каналов (watcher) с REST-управлением
- Динамический список полей извлечения (schema), задаётся через API
- Docker Compose деплой (backend + Mongo)

Быстрый старт (docker-compose)
1) Создайте папки и положите user session:
```
mkdir -p sessions data/images data/exports
# положите сюда sessions/user.session
```
2) Запустите:
```
docker-compose up --build -d
```
3) Проверьте здоровье:
```
curl http://localhost:8001/api/health
```

Управление каналами
- GET /api/channels -> {channels}
- POST /api/channels/set {"channels":["arendakv_msk","another_channel"]}
- POST /api/channels/add {"channel":"arendakv_msk"}
- POST /api/channels/remove {"channel":"arendakv_msk"}

Управление watcher
- POST /api/watcher/start
- POST /api/watcher/stop
- GET  /api/watcher/status
- POST /api/watcher/run_once (однократный обход всех каналов)

Схема динамических полей
- GET /api/schema -> {fields: [...]}
- POST /api/schema {"fields":[{"key":"deposit","question":"Какой залог?","type":"number"}]}
  - type: string|number|phone, regex (опционально) — если указан, сначала применяется regex, затем QA

Парсинг и CSV
- POST /api/parse {"channel":"arendakv_msk","limit":30}
- POST /api/parse_to_csv {"channel":"arendakv_msk","limit":30,"csv_name":"rent.csv"}

Переменные окружения (важное)
- MONGO_URL=mongodb://mongo:27017/tg_parser
- MONGO_DB_NAME=tg_parser
- PYROGRAM_SESSION_DIR=/sessions
- PYROGRAM_SESSION_NAME=user
- IMAGES_DIR=/data/images
- EXPORTS_DIR=/data/exports
- WATCH_INTERVAL=60, PER_CHANNEL_DELAY=2.0
- NOMINATIM_EMAIL=you@example.com (желательно)
- RU_QA_MODEL=DeepPavlov/rubert-base-cased-squad (по умолчанию)
- TG_API_ID/TG_API_HASH — если сессии этого требуют

Примечания
- Для корректной работы Pyrogram используйте валидную user session в папке sessions.
- Watcher хранит last_message_id в коллекции state.
- Индекс в posts: уникальная пара (channel, message_id).