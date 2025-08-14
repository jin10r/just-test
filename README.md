# Telegram Rental Parser (Watcher + Dynamic Schema + Autostart)

Фоновый парсер, который нон-стоп мониторит список телеграм-каналов и наполняет базу данных. Поддерживает динамическую схему полей, BERT QA для неструктурированных caption, до 3 фото, геокод через Nominatim. Готов к деплою через docker-compose.

Запуск
1) Подготовить каталоги и user session:
```
mkdir -p sessions data/images data/exports
# положите сюда sessions/user.session
```
2) Поднять:
```
docker-compose up --build -d
```
3) Watcher стартует автоматически (WATCH_AUTOSTART=true).

Управление
- Каналы: GET /api/channels; POST /api/channels/set|add|remove
- Схема: GET /api/schema; POST /api/schema
- Watcher: POST /api/watcher/start|stop; GET /api/watcher/status; POST /api/watcher/run_once

Переменные окружения
- MONGO_URL=mongodb://mongo:27017/tg_parser
- WATCH_AUTOSTART=true — автозапуск фонового наблюдателя
- WATCH_INTERVAL=60, PER_CHANNEL_DELAY=2.0
- PYROGRAM_SESSION_DIR=/sessions, PYROGRAM_SESSION_NAME=user
- IMAGES_DIR=/data/images, EXPORTS_DIR=/data/exports
- NOMINATIM_EMAIL=you@example.com (рекомендуется)
- RU_QA_MODEL=DeepPavlov/rubert-base-cased-squad (по умолчанию)

Как это работает
- Список каналов хранится в Mongo (config, _id="channels").
- Состояние последнего поста по каналу — в коллекции state (_id="last:<channel>").
- В фоне watcher обходит каналы с заданным интервалом и обрабатывает только новые посты, сохраняя распарсенные документы в коллекции posts (upsert по (channel, message_id)).
- Извлечение параметров: правила + BERT QA (fallback) + настраиваемые поля через /api/schema.