import os
import re
import uuid
import asyncio
import time
import logging
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
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
    # Allow running even if .env not present; user will set in production
    logger.warning("MONGO_URL not set. Set backend/.env with MONGO_URL for persistence.")

MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "tg_parser")  # env-driven per requirements

YANDEX_GEOCODER_API_KEY = (
    os.environ.get("YANDEX_GEOCODER_API_KEY")
    or os.environ.get("YA_GEOCODER_API_KEY")
)

SESSION_DIR = os.environ.get("PYROGRAM_SESSION_DIR", "/app/sessions")
SESSION_NAME = os.environ.get("PYROGRAM_SESSION_NAME", "user")

IMAGES_DIR = os.environ.get("IMAGES_DIR", "/app/data/images")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# Cache heavy models lazily
_ner_pipeline = None
_morph = MorphVocab()
_addr_extractor = AddrExtractor(_morph)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def get_db():
    if not MONGO_URL:
        raise RuntimeError("MONGO_URL is not configured")
    client = AsyncIOMotorClient(MONGO_URL)
    return client[MONGO_DB_NAME]


def load_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        # cointegrated/rubert-tiny-ner is fast and good enough
        logger.info("Loading ruBERT NER pipeline (cointegrated/rubert-tiny-ner)...")
        _ner_pipeline = pipeline(
            "token-classification",
            model="cointegrated/rubert-tiny-ner",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


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
# Parsing helpers
# ------------------------------------------------------------
PHONE_RE = re.compile(r"(\+7|8)\s*[\(\-\s]?\d{3}[\)\-\s]?\s?\d{3}[\-\s]?\d{2}[\-\s]?\d{2}")
PRICE_RE = re.compile(r"(?:(?:цена|стоимость|арендная\s*плата)[:\s]*)?(\d[\d\s]{3,})(?:\s*(?:₽|руб(?:\.|лей|ля)?|р\.?))?", re.IGNORECASE)
METRO_RE = re.compile(r"(?:\bм\.?|метро)\s*[:\-]?\s*([А-ЯЁа-яё\- ]{2,})")


def clean_number(num: str) -> int:
    return int(re.sub(r"\D", "", num)) if num else None


def extract_address_with_natasha(text: str) -> Optional[str]:
    matches = list(_addr_extractor.find(text))
    if not matches:
        return None
    # Choose the longest span
    best = max(matches, key=lambda m: len(m.span))
    return text[best.span.start : best.span.stop]


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
        }

    # Metro
    m = METRO_RE.search(text)
    if m:
        metro = m.group(1).strip().strip("-:.")

    # Phone
    p = PHONE_RE.search(text)
    if p:
        phone = re.sub(r"\D", "", p.group(0))  # normalized digits

    # Price
    pr = PRICE_RE.search(text)
    if pr:
        price_val = clean_number(pr.group(1))

    # Address via Natasha first
    address = extract_address_with_natasha(text)

    # Fallback: rule-based – pick line containing address markers
    if not address:
        for line in text.splitlines():
            if re.search(r"(ул\.|улица|пр\-кт|проспект|просп\.|бульвар|бул\.|переулок|пер\.|шоссе|ш\.|дом|д\.)", line, re.IGNORECASE):
                address = line.strip()
                break

    # Description: remove extracted items from text roughly
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
    }


def geocode_nominatim(address: str) -> Optional[Dict[str, float]]:
    if not address:
        return None
    if not YANDEX_GEOCODER_API_KEY:
        logger.warning("YANDEX_GEOCODER_API_KEY not configured. Skipping geocoding.")
        return None
    try:
        url = "https://geocode-maps.yandex.ru/1.x/"
        params = {
            "apikey": YANDEX_GEOCODER_API_KEY,
            "geocode": address,
            "format": "json",
            "lang": "ru_RU",
            "rspn": 0,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        feature_members = data.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
        if not feature_members:
            return None
        first = feature_members[0]["GeoObject"]
        pos = first["Point"]["pos"].split()
        lon, lat = map(float, pos)
        return {"lat": lat, "lon": lon}
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return None


async def ensure_indexes(db):
    await db.posts.create_index([("channel", 1), ("message_id", 1)], unique=True)


async def parse_channel(channel: str, limit: int = 30) -> Dict[str, Any]:
    """Parse latest messages from a public channel and store structured posts."""
    if not channel:
        raise ValueError("channel is required")

    db = get_db()
    await ensure_indexes(db)

    client = get_pyrogram_client()

    inserted = 0
    updated = 0
    errors: List[str] = []

    async with client:
        try:
            async for msg in client.get_chat_history(channel, limit=limit):
                try:
                    # Determine text
                    text = (msg.caption or msg.text or "").strip()

                    # If this message belongs to media group, get the group to find first photo + caption
                    photo_path = None
                    if msg.media_group_id:
                        try:
                            grp = await client.get_media_group(chat_id=msg.chat.id, message_id=msg.id)
                            # choose first photo in group
                            for m in grp:
                                if m.photo:
                                    photo_path = await client.download_media(m, file_name=os.path.join(IMAGES_DIR, f"{uuid.uuid4()}.jpg"))
                                    break
                            # prefer caption from any message in the group
                            if not text:
                                for m in grp:
                                    if (m.caption or m.text):
                                        text = (m.caption or m.text).strip()
                                        break
                        except Exception as e:
                            logger.warning(f"Failed to process media group {msg.media_group_id}: {e}")
                    else:
                        if msg.photo:
                            try:
                                photo_path = await client.download_media(msg, file_name=os.path.join(IMAGES_DIR, f"{uuid.uuid4()}.jpg"))
                            except Exception as e:
                                logger.warning(f"Photo download error for message {msg.id}: {e}")

                    # Extract parameters
                    base = extract_info(text)

                    # ruBERT NER (optional enrichment) - try to find additional LOC tokens if address is missing
                    try:
                        if not base.get("address") and text:
                            ner = load_ner_pipeline()
                            ents = ner(text)
                            locs = [e["word"] for e in ents if e.get("entity_group") in {"LOC", "PER", "ORG", "MISC"}]
                            if locs:
                                # Join consecutive tokens into string, simple heuristic
                                base["address"] = " ".join(locs)
                    except Exception as e:
                        logger.warning(f"NER enrichment failed: {e}")

                    coords = geocode_nominatim(base.get("address")) if base.get("address") else None

                    doc = {
                        "_id": str(uuid.uuid4()),
                        "channel": channel,
                        "message_id": msg.id,
                        "date": msg.date.isoformat() if msg.date else None,
                        "text": text or None,
                        "photo_path": photo_path,
                        "metro": base.get("metro"),
                        "address": base.get("address"),
                        "coords": coords,
                        "phone": base.get("phone"),
                        "price": base.get("price"),
                        "currency": "RUB" if base.get("price") else None,
                        "description": base.get("description"),
                    }

                    # Upsert by channel+message_id
                    res = await db.posts.update_one(
                        {"channel": channel, "message_id": msg.id},
                        {"$set": doc},
                        upsert=True,
                    )
                    if res.upserted_id:
                        inserted += 1
                    else:
                        updated += res.modified_count
                except Exception as e:
                    logger.exception(f"Message {msg.id} processing failed: {e}")
                    errors.append(str(e))
        except RPCError as e:
            raise HTTPException(status_code=400, detail=f"Telegram error: {e}")

    return {"inserted": inserted, "updated": updated, "errors": errors}


# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------
app = FastAPI(title="Telegram Rental Parser", version="1.0.0")


class ParseRequest(BaseModel):
    channel: str
    limit: int = 30


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/parse")
async def api_parse(req: ParseRequest):
    result = await parse_channel(req.channel, req.limit)
    return result


@app.get("/api/posts")
async def list_posts(
    channel: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200)
):
    db = get_db()
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