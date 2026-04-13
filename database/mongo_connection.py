import os
from functools import lru_cache
from pathlib import Path

import yaml
from pymongo import MongoClient


def _load_db_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config.get("database", {}) or {}


@lru_cache(maxsize=1)
def _get_client_and_db_name() -> tuple[MongoClient, str]:
    db_cfg = _load_db_config()
    uri = os.getenv("MONGODB_URI", db_cfg.get("mongodb_uri"))
    db_name = os.getenv("MONGODB_DB_NAME", db_cfg.get("db_name", "travel_ir"))
    if not uri:
        raise ValueError("Missing MongoDB URI. Set MONGODB_URI or database.mongodb_uri in config.")

    client = MongoClient(
        uri,
        connectTimeoutMS=30000,
        socketTimeoutMS=60000,
        serverSelectionTimeoutMS=30000,
        heartbeatFrequencyMS=30000,
        retryWrites=True,
        retryReads=True,
    )
    return client, db_name


def get_database():
    client, db_name = _get_client_and_db_name()
    return client[db_name]


def get_collection_names() -> dict:
    db_cfg = _load_db_config()
    collections = db_cfg.get("collections", {}) or {}
    return {
        "places": str(collections.get("places", "places")),
        "reviews": str(collections.get("reviews", "reviews")),
    }
