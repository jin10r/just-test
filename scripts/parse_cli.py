#!/usr/bin/env python3
import asyncio
import argparse
from backend.server import parse_channel


def main():
    parser = argparse.ArgumentParser(description="Parse Telegram channel posts and store to MongoDB")
    parser.add_argument("channel", help="Channel username or link, e.g. arendakv_msk or https://t.me/arendakv_msk")
    parser.add_argument("--limit", type=int, default=30, help="How many latest messages to parse")
    args = parser.parse_args()

    # Normalize channel
    ch = args.channel
    if ch.startswith("https://t.me/"):
        ch = ch.split("https://t.me/")[-1].strip("/")

    result = asyncio.run(parse_channel(ch, args.limit))
    print(result)


if __name__ == "__main__":
    main()