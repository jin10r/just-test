import os
import re
import uuid
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List

import csv
import requests
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient

# Pyrogram (Telegram)
from pyrogram import Client
from pyrogram.errors import RPCError

# NLP: BERT NER (ruBERT tiny) and Address extraction (Natasha)
from transformers import pipeline
from natasha import AddrExtractor, MorphVocab

# ------------------------------------------------------------
# Environment & Globals
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tg_parser")

MONGO_URL = os.environ.get("MONGO_URL")
if not MONGO_URL:
    logger.warning("MONGO_URL not set. DB writes will be skipped; data can be exported to CSV.")

MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "roommate_app")  # env-driven to match your DB

SESSION_DIR = os.environ.get("PYROGRAM_SESSION_DIR", "/app/sessions")
SESSION_NAME = os.environ.get("PYROGRAM_SESSION_NAME", "user")

IMAGES_DIR = os.environ.get("IMAGES_DIR", "/app/data/images")
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "/app/data/exports")
WATCH_INTERVAL = int(os.environ.get("WATCH_INTERVAL", "60"))
PER_CHANNEL_DELAY = float(os.environ.get("PER_CHANNEL_DELAY", "2.0"))
WATCH_AUTOSTART = os.environ.get("WATCH_AUTOSTART", "false").lower() in {"1", "true", "yes"}

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Cache heavy models lazily
_ner_pipeline = None
_qa_pipeline = None
_morph = MorphVocab()
_addr_extractor = AddrExtractor(_morph)

WATCH_TASK: Optional[asyncio.Task] = None
WATCH_RUNNING: bool = False


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def maybe_get_db():
    if not MONGO_URL:
        return None
    client = AsyncIOMotorClient(MONGO_URL)
    return client[MONGO_DB_NAME]


def load_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        logger.info("Loading ruBERT NER pipeline (cointegrated/rubert-tiny-ner)...")
        _ner_pipeline = pipeline(
            "token-classification",
            model="cointegrated/rubert-tiny-ner",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def load_qa_pipeline():
    """Russian QA for unstructured caption parsing.
    Prefer Russian-finetuned QA model. Falls back to multilingual XLM-R if not available.
    """
    global _qa_pipeline
    if _qa_pipeline is None:
        model_name = os.environ.get("RU_QA_MODEL", "DeepPavlov/rubert-base-cased-squad")
        try:
            logger.info(f"Loading QA pipeline: {model_name}")
            _qa_pipeline = pipeline("question-answering", model=model_name)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to xlm-roberta-large-finetuned-russian-squad: {e}")
            _qa_pipeline = pipeline("question-answering", model="xlm-roberta-large-finetuned-russian-squad")
    return _qa_pipeline


def get_pyrogram_client() -> Client:
    """Create Client using existing user .session. api_id/hash are optional.

    If Pyrogram requires api_id/api_hash, we try to read TG_API_ID/TG_API_HASH envs.
    """
    api_id = os.environ.get("TG_API_ID")
    api_hash = os.environ.get("TG_API_HASH")

    try:
        if api_id and api_hash:
            logger.info("Initializing Pyrogram with provided TG_API_ID/TG_API_HASH and existing session...")
            return Client(SESSION_NAME, workdir=SESSION_DIR, api_id=int(api_id), api_hash=api_hash, no_updates=True)
        else:
            logger.info("Initializing Pyrogram with existing session only (no api_id/api_hash).")
            return Client(SESSION_NAME, workdir=SESSION_DIR, no_updates=True)
    except Exception as e:
        logger.error(f"Failed to create Pyrogram Client: {e}")
        raise


# ------------------------------------------------------------
# Config storage (channels, schema, state)
# ------------------------------------------------------------

async def cfg_get_channels(db) -> List[str]:
    if not db:
        return []
    doc = await db.config.find_one({"_id": "channels"})
    return (doc or {}).get("list", [])


async def cfg_set_channels(db, channels: List[str]):
    await db.config.update_one({"_id": "channels"}, {"$set": {"list": channels}}, upsert=True)


async def cfg_add_channel(db, ch: str):
    arr = await cfg_get_channels(db)
    if ch not in arr:
        arr.append(ch)
        await cfg_set_channels(db, arr)


async def cfg_remove_channel(db, ch: str):
    arr = await cfg_get_channels(db)
    arr = [x for x in arr if x != ch]
    await cfg_set_channels(db, arr)


async def cfg_get_schema(db) -> List[Dict[str, Any]]:
    if not db:
        return []
    doc = await db.config.find_one({"_id": "schema"})
    return (doc or {}).get("fields", [])


async def cfg_set_schema(db, fields: List[Dict[str, Any]]):
    await db.config.update_one({"_id": "schema"}, {"$set": {"fields": fields}}, upsert=True)


async def state_get_last_id(db, channel: str) -> int:
    if not db:
        return 0
    doc = await db.state.find_one({"_id": f"last:{channel}"})
    return int((doc or {}).get("val", 0))


async def state_set_last_id(db, channel: str, last_id: int):
    if not db:
        return
    await db.state.update_one({"_id": f"last:{channel}"}, {"$set": {"val": int(last_id)}}, upsert=True)


# ------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------
PHONE_RE = re.compile(r"(\+7|8)\s*[\(\-\s]?\d{3}[\)\-\s]?\s?\d{3}[\-\s]?\d{2}[\-\s]?\d{2}")
PRICE_RE = re.compile(r"(?:(?:цена|стоимость|арендная\s*плата)[:\s]*)?(\d[\d\s]{3,})(?:\s*(?:₽|руб(?:\.|лей|ля)?|р\.?))?", re.IGNORECASE)
METRO_RE = re.compile(r"(?:\bм\.?|метро)\s*[:\-]?\s*([А-ЯЁа-яё\- ]{2,})")
ROOMS_RE_NUM = re.compile(r"(\d+)\s*[- ]?(?:к(?!\w)|комнатн)\w*", re.IGNORECASE)
# word forms
ROOMS_WORDS = {
    "однокомнат": 1,
    "двухкомнат": 2,
    "трехкомнат": 3,
    "трёхкомнат": 3,
    "четырехкомнат": 4,
    "четырёхкомнат": 4,
    "пятикомнат": 5,
}
# floor patterns
FLOOR_PAIR_RE = re.compile(r"(\d+)\s*[/\\|]\s*(\d+)\s*этаж", re.IGNORECASE)
FLOOR_IN_RE = re.compile(r"(?:этаж[:\s]*|на\s+)(\d+)(?:\s*(?:из|/|\\)\s*(\d+))?\s*этаж?", re.IGNORECASE)


def clean_number(num: str) -> Optional[int]:
    try:
        return int(re.sub(r"\D", "", num)) if num else None
    except Exception:
        return None


def extract_address_with_natasha(text: str) -> Optional[str]:
    matches = list(_addr_extractor.find(text))
    if not matches:
        return None
    best = max(matches, key=lambda m: len(m.span))
    return text[best.span.start : best.span.stop]


def extract_rooms(text: str) -> Optional[int]:
    if not text:
        return None
    m = ROOMS_RE_NUM.search(text)
    if m:
        return clean_number(m.group(1))
    low = text.lower()
    for key, val in ROOMS_WORDS.items():
        if key in low:
            return val
    m2 = re.search(r"(\d+)\s*кк?\b", low)
    if m2:
        return clean_number(m2.group(1))
    return None


def extract_floor(text: str) -> Dict[str, Optional[int]]:
    if not text:
        return {"floor": None, "floors_total": None}
    m = FLOOR_PAIR_RE.search(text)
    if m:
        return {"floor": clean_number(m.group(1)), "floors_total": clean_number(m.group(2))}
    m2 = FLOOR_IN_RE.search(text)
    if m2:
        return {"floor": clean_number(m2.group(1)), "floors_total": clean_number(m2.group(2)) if m2.group(2) else None}
    m3 = re.search(r"этаж\s*(\d+)", text, re.IGNORECASE)
    if m3:
        return {"floor": clean_number(m3.group(1)), "floors_total": None}
    return {"floor": None, "floors_total": None}


def qa(text: str, question: str) -> Optional[str]:
    if not text:
        return None
    try:
        qa_pipe = load_qa_pipeline()
        res = qa_pipe(question=question, context=text)
        ans = (res.get("answer") or "").strip()
        if ans:
            return ans
    except Exception as e:
        logger.warning(f"QA failed for '{question}': {e}")
    return None


def extract_dynamic(text: str, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generic field extractor driven by schema: [{key, question, type, regex?}]"""
    out: Dict[str, Any] = {}
    if not text:
        return out
    for f in fields or []:
        key = f.get("key")
        if not key:
            continue
        val = None
        pattern = f.get("regex")
        if pattern:
            try:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    val = m.group(1) if m.groups() else m.group(0)
            except re.error:
                pass
        if val is None and f.get("question"):
            val = qa(text, f["question"]) or None
        t = (f.get("type") or "string").lower()
        if val is not None:
            if t == "number":
                val = clean_number(str(val))
            elif t == "phone":
                val = re.sub(r"\D", "", str(val)) or None
            else:
                val = str(val).strip()
        out[key] = val
    return out


def extract_info(text: str) -> Dict[str, Any]:
    metro = None
    phone = None
    price_val: Optional[int] = None

    if not text:
        return {
            "metro": metro,
            "address": None,
            "phone": phone,
            "price": price_val,
            "description": None,
            "rooms": None,
            "floor": None,
            "floors_total": None,
        }

    m = METRO_RE.search(text)
    if m:
        metro = m.group(1).strip().strip("-:.")
    if not metro:
        maybe = qa(text, "Какая станция метро?")
        if maybe and len(maybe) <= 40:
            metro = maybe

    p = PHONE_RE.search(text)
    if p:
        phone = re.sub(r"\D", "", p.group(0))
    if not phone:
        maybe_phone = qa(text, "Какой номер телефона?") or qa(text, "Контактный телефон?")
        if maybe_phone:
            phone = re.sub(r"\D", "", maybe_phone)
            if not phone:
                phone = None

    pr = PRICE_RE.search(text)
    if pr:
        price_val = clean_number(pr.group(1))
    if price_val is None:
        maybe_price = qa(text, "Какая цена аренды?") or qa(text, "Арендная плата в месяц?")
        if maybe_price:
            price_val = clean_number(maybe_price)

    address = extract_address_with_natasha(text)
    if not address:
        maybe_addr = qa(text, "Какой адрес?") or qa(text, "Укажите адрес объекта")
        if maybe_addr and len(maybe_addr) >= 6:
            address = maybe_addr

    if not address:
        for line in text.splitlines():
            if re.search(r"(ул\.|улица|пр\-кт|проспект|просп\.|бульвар|бул\.|переулок|пер\.|шоссе|ш\.|дом|д\.)", line, re.IGNORECASE):
                address = line.strip()
                break

    rooms = extract_rooms(text)
    if rooms is None:
        maybe_rooms = qa(text, "Сколько комнат?") or qa(text, "Количество комнат?")
        if maybe_rooms:
            rooms = clean_number(maybe_rooms)

    floor_info = extract_floor(text)
    if floor_info["floor"] is None:
        maybe_floor = qa(text, "Какой этаж?")
        if maybe_floor:
            floor_info["floor"] = clean_number(maybe_floor)

    desc = text
    if m:
        desc = desc.replace(m.group(0), "")
    if p:
        desc = desc.replace(p.group(0), "")
    if pr:
        desc = desc.replace(pr.group(0), "")
    if address:
        desc = desc.replace(address, "")
    desc = re.sub(r"\n{2,}", "\n", desc).strip()

    return {
        "metro": metro,
        "address": address,
        "phone": phone,
        "price": price_val,
        "description": desc or None,
        "rooms": rooms,
        "floor": floor_info["floor"],
        "floors_total": floor_info["floors_total"],
    }


# ---------- Nominatim Geocoding ----------

def geocode_nominatim(address: str) -> Optional[Dict[str, float]]:
    if not address:
        return None
    email = os.environ.get("NOMINATIM_EMAIL")
    headers = {
        "User-Agent": f"tg_parser/1.0 ({email})" if email else "tg_parser/1.0 (contact: set NOMINATIM_EMAIL)",
        "Accept-Language": "ru",
    }
    params = {"q": address, "format": "json", "limit": 1, "addressdetails": 1}
    delay = 1.0
    for attempt in range(3):
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                logger.warning("Nominatim rate limited (429). Backing off...")
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            first = data[0]
            lat = float(first.get("lat"))
            lon = float(first.get("lon"))
            return {"lat": lat, "lon": lon}
        except Exception as e:
            logger.warning(f"Nominatim request failed (attempt {attempt+1}): {e}")
            time.sleep(delay)
            delay *= 2
    logger.error("Nominatim geocoding failed after retries")
    return None


async def ensure_indexes(db):
    if not db:
        return
    # ensure auxiliary collection posts still exists if used elsewhere
    await db.posts.create_index([("channel", 1), ("message_id", 1)], unique=True)
    # properties indexes are created by your init script; we can add upsert safeguards here if needed


# ------------------------------------------------------------
# Core parsing for a single message
# ------------------------------------------------------------
async def process_message(channel: str, msg, client: Client, db) -> Dict[str, Any]:
    text = (msg.caption or msg.text or "").strip()

    # Collect up to 3 photos
    photo_paths: List[str] = []
    if msg.media_group_id:
        try:
            grp = await client.get_media_group(chat_id=msg.chat.id, message_id=msg.id)
            for m in grp:
                if m.photo:
                    path = await client.download_media(m, file_name=os.path.join(IMAGES_DIR, f"{uuid.uuid4()}.jpg"))
                    if path:
                        photo_paths.append(path)
                    if len(photo_paths) >= 3:
                        break
            if not text:
                for m in grp:
                    if (m.caption or m.text):
                        text = (m.caption or m.text).strip()
                        break
        except Exception as e:
            logger.warning(f"Failed to process media group {msg.media_group_id}: {e}")
    else:
        if getattr(msg, "photo", None):
            try:
                path = await client.download_media(msg, file_name=os.path.join(IMAGES_DIR, f"{uuid.uuid4()}.jpg"))
                if path:
                    photo_paths.append(path)
            except Exception as e:
                logger.warning(f"Photo download error for message {msg.id}: {e}")

    base = extract_info(text)

    # NER enrichment if address missing
    try:
        if not base.get("address") and text:
            ner = load_ner_pipeline()
            ents = ner(text)
            locs = [e["word"] for e in ents if e.get("entity_group") in {"LOC"}]
            if locs:
                base["address"] = " ".join(locs)
    except Exception as e:
        logger.warning(f"NER enrichment failed: {e}")

    coords = geocode_nominatim(base.get("address")) if base.get("address") else None

    dyn_fields = await cfg_get_schema(db) if db else []
    extras = extract_dynamic(text, dyn_fields)

    # Build property document according to your properties schema
    title = base.get("address") or base.get("metro") or (text[:140] if text else None)
    if coords:
        location = {"type": "Point", "coordinates": [coords["lon"], coords["lat"]]}
    else:
        location = None

    property_doc = {
        "id": str(uuid.uuid4()),
        "title": title or "Объявление",
        "price": base.get("price") or 0,
        "location": location or {"type": "Point", "coordinates": [0.0, 0.0]},
        # optional fields we add (ваши валидаторы их допустят)
        "metro_station": base.get("metro"),
        "address": base.get("address"),
        "phone": base.get("phone"),
        "rooms": base.get("rooms"),
        "floor": base.get("floor"),
        "floors_total": base.get("floors_total"),
        "description": base.get("description"),
        "photos": photo_paths,
        "channel": channel,
        "message_id": msg.id,
        **extras,
        "created_at": msg.date.isoformat() if msg.date else None,
    }

    if db:
        # Write into properties collection as required
        res = await db.properties.update_one({"channel": channel, "message_id": msg.id}, {"$set": property_doc}, upsert=True)
        return {"doc": property_doc, "inserted": int(bool(res.upserted_id)), "updated": res.modified_count}
    else:
        return {"doc": property_doc, "inserted": 0, "updated": 0}


# ------------------------------------------------------------
# Batch parsing helpers
# ------------------------------------------------------------
async def parse_channel(channel: str, limit: int = 30) -> Dict[str, Any]:
    if not channel:
        raise ValueError("channel is required")

    db = maybe_get_db()
    await ensure_indexes(db)

    client = get_pyrogram_client()

    inserted = 0
    updated = 0
    errors: List[str] = []
    items: List[Dict[str, Any]] = []

    async with client:
        try:
            async for msg in client.get_chat_history(channel, limit=limit):
                try:
                    res = await process_message(channel, msg, client, db)
                    items.append(res["doc"])
                    inserted += res.get("inserted", 0)
                    updated += res.get("updated", 0)
                except Exception as e:
                    logger.exception(f"Message {getattr(msg,'id',None)} processing failed: {e}")
                    errors.append(str(e))
        except RPCError as e:
            raise HTTPException(status_code=400, detail=f"Telegram error: {e}")

    return {"inserted": inserted, "updated": updated, "errors": errors, "items": items}


async def parse_new_posts(channel: str, batch_limit: int = 100) -> Dict[str, Any]:
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is required for watcher mode")

    await ensure_indexes(db)

    last_id = await state_get_last_id(db, channel)
    client = get_pyrogram_client()

    inserted = 0
    updated = 0
    errors: List[str] = []
    processed_max_id = last_id

    async with client:
        msgs: List[Any] = []
        async for m in client.get_chat_history(channel, limit=batch_limit):
            if m.id and m.id > last_id:
                msgs.append(m)
        msgs = list(sorted(msgs, key=lambda x: x.id))
        for msg in msgs:
            try:
                res = await process_message(channel, msg, client, db)
                inserted += res.get("inserted", 0)
                updated += res.get("updated", 0)
                processed_max_id = max(processed_max_id, msg.id)
            except Exception as e:
                errors.append(str(e))

    if processed_max_id > last_id:
        await state_set_last_id(db, channel, processed_max_id)

    return {"inserted": inserted, "updated": updated, "errors": errors, "last_id": processed_max_id}


async def watcher_loop():
    global WATCH_RUNNING
    WATCH_RUNNING = True
    logger.info("Watcher loop started")
    while WATCH_RUNNING:
        try:
            db = maybe_get_db()
            if not db:
                logger.warning("Watcher running without DB configured. Sleeping...")
                await asyncio.sleep(WATCH_INTERVAL)
                continue
            channels = await cfg_get_channels(db)
            for ch in channels:
                try:
                    await parse_new_posts(ch)
                except Exception as e:
                    logger.warning(f"Watcher error on {ch}: {e}")
                await asyncio.sleep(PER_CHANNEL_DELAY)
        except Exception as e:
            logger.exception(f"Watcher loop error: {e}")
        await asyncio.sleep(WATCH_INTERVAL)
    logger.info("Watcher loop stopped")


# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------
app = FastAPI(title="Telegram Rental Parser", version="2.1.0")


class ParseRequest(BaseModel):
    channel: str
    limit: int = 30


class ParseToCSVRequest(ParseRequest):
    csv_name: Optional[str] = None  # optional custom filename


class ChannelsPayload(BaseModel):
    channels: List[str] = Field(default_factory=list)


class ChannelOne(BaseModel):
    channel: str


class SchemaPayload(BaseModel):
    fields: List[Dict[str, Any]] = Field(default_factory=list)


@app.on_event("startup")
async def _maybe_start_watcher():
    if WATCH_AUTOSTART:
        db = maybe_get_db()
        if db:
            logger.info("WATCH_AUTOSTART is enabled. Starting watcher...")
            await ensure_indexes(db)
            global WATCH_TASK, WATCH_RUNNING
            if not (WATCH_TASK and not WATCH_TASK.done() and WATCH_RUNNING):
                WATCH_RUNNING = True
                WATCH_TASK = asyncio.create_task(watcher_loop())
        else:
            logger.warning("WATCH_AUTOSTART is on but MongoDB is not configured. Skipping watcher start.")


@app.get("/api/health")
async def health():
    running = bool(WATCH_TASK and not WATCH_TASK.done() and WATCH_RUNNING)
    return {"status": "ok", "watcher_running": running}


@app.post("/api/parse")
async def api_parse(req: ParseRequest):
    result = await parse_channel(req.channel, req.limit)
    return {"inserted": result["inserted"], "updated": result["updated"], "errors": result["errors"], "count": len(result["items"]) }


@app.post("/api/parse_to_csv")
async def api_parse_to_csv(req: ParseToCSVRequest):
    result = await parse_channel(req.channel, req.limit)
    items = result["items"]
    fname = req.csv_name or f"{req.channel}_{int(time.time())}.csv"
    csv_path = os.path.join(EXPORTS_DIR, fname)

    fieldnames = [
        "channel", "message_id", "date",
        "photos",
        "metro", "address", "price", "phone",
        "rooms", "floor", "description"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            writer.writerow({
                "channel": it.get("channel"),
                "message_id": it.get("message_id"),
                "date": it.get("date"),
                "photos": "; ".join((it.get("photo_paths") or [])[:3]),
                "metro": it.get("metro"),
                "address": it.get("address"),
                "price": it.get("price"),
                "phone": it.get("phone"),
                "rooms": it.get("rooms"),
                "floor": it.get("floor"),
                "description": it.get("description"),
            })

    return {
        "csv_path": csv_path,
        "inserted": result["inserted"],
        "updated": result["updated"],
        "count": len(items),
        "errors": result["errors"],
    }


# ---- Channels management ----
@app.get("/api/channels")
async def api_channels_list():
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    return {"channels": await cfg_get_channels(db)}


@app.post("/api/channels/set")
async def api_channels_set(payload: ChannelsPayload):
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    await cfg_set_channels(db, payload.channels)
    return {"ok": True, "channels": await cfg_get_channels(db)}


@app.post("/api/channels/add")
async def api_channels_add(payload: ChannelOne):
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    await cfg_add_channel(db, payload.channel)
    return {"ok": True, "channels": await cfg_get_channels(db)}


@app.post("/api/channels/remove")
async def api_channels_remove(payload: ChannelOne):
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    await cfg_remove_channel(db, payload.channel)
    return {"ok": True, "channels": await cfg_get_channels(db)}


# ---- Schema management ----
@app.get("/api/schema")
async def api_schema_get():
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    return {"fields": await cfg_get_schema(db)}


@app.post("/api/schema")
async def api_schema_set(payload: SchemaPayload):
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    await cfg_set_schema(db, payload.fields)
    return {"ok": True}


# ---- Watcher controls ----
@app.post("/api/watcher/start")
async def api_watcher_start(background_tasks: BackgroundTasks):
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    global WATCH_TASK, WATCH_RUNNING
    if WATCH_TASK and not WATCH_TASK.done() and WATCH_RUNNING:
        return {"running": True}
    WATCH_RUNNING = True
    WATCH_TASK = asyncio.create_task(watcher_loop())
    return {"running": True}


@app.post("/api/watcher/stop")
async def api_watcher_stop():
    global WATCH_RUNNING
    WATCH_RUNNING = False
    return {"running": False}


@app.get("/api/watcher/status")
async def api_watcher_status():
    db = maybe_get_db()
    channels = await cfg_get_channels(db) if db else []
    running = bool(WATCH_TASK and not WATCH_TASK.done() and WATCH_RUNNING)
    return {"running": running, "interval": WATCH_INTERVAL, "channels": channels}


@app.post("/api/watcher/run_once")
async def api_watcher_run_once():
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured")
    channels = await cfg_get_channels(db)
    results = {}
    for ch in channels:
        results[ch] = await parse_new_posts(ch)
        await asyncio.sleep(PER_CHANNEL_DELAY)
    return {"results": results}


@app.get("/api/posts")
async def list_posts(
    channel: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200)
):
    db = maybe_get_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB is not configured in this environment")
    query: Dict[str, Any] = {}
    if channel:
        query["channel"] = channel
    cursor = db.posts.find(query).sort("date", -1).limit(limit)
    items: List[Dict[str, Any]] = []
    async for item in cursor:
        item["id"] = item.pop("_id")
        items.append(item)
    return {"items": items}


# IMPORTANT: The server must bind to 0.0.0.0:8001 via supervisor (configured outside).
# Do not run uvicorn here.