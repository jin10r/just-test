# Telegram Rental Parser (Pyrogram + ruBERT + FastAPI)

This service parses Telegram channel posts (e.g. rentals in Moscow) and extracts:
- Metro station
- Address (with geocoding to coordinates)
- Phone
- Description
- Price
- One photo per post (downloaded)

It uses:
- Pyrogram (works with an existing authorized user .session file)
- ruBERT tiny NER for Russian (cointegrated/rubert-tiny-ner)
- Natasha for address extraction
- FastAPI API, MongoDB for storage
- Yandex Geocoder API for coordinates

## Environment variables
- MONGO_URL: Mongo connection string (required)
- MONGO_DB_NAME: Database name (default: tg_parser)
- PYROGRAM_SESSION_DIR: Path to folder with the user session (default: /app/sessions)
- PYROGRAM_SESSION_NAME: Session file base name without extension (default: user)
- TG_API_ID / TG_API_HASH: Optional. Only used if Pyrogram requires app credentials to open the existing session
- YANDEX_GEOCODER_API_KEY (or YA_GEOCODER_API_KEY): for geocoding
- IMAGES_DIR: Where to save images (default: /app/data/images)

## API (binds via supervisor to 0.0.0.0:8001)
- GET /api/health
- POST /api/parse {"channel": "arendakv_msk", "limit": 30}
- GET /api/posts?channel=arendakv_msk&limit=20

## CLI
```
python3 scripts/parse_cli.py https://t.me/arendakv_msk --limit 30
```

## Notes
- Place your existing Pyrogram user session file at `${PYROGRAM_SESSION_DIR}/${PYROGRAM_SESSION_NAME}.session`.
- API ID/Hash are not required if the session file allows login without them; otherwise provide TG_API_ID/TG_API_HASH as envs.
- Yandex Geocoder key is required in production to get coordinates.