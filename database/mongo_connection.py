from pathlib import Path

import yaml
from pymongo import MongoClient


def get_database():
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    db_cfg = config.get("database", {})
    uri = db_cfg.get("mongodb_uri")
    db_name = db_cfg.get("db_name", "travel_ir")
    if not uri:
        raise ValueError("Missing mongodb_uri in config.yaml")
    
    # Tăng timeout + retry + heartbeat cho kết nối ổn định hơn
    client = MongoClient(
        uri, 
        connectTimeoutMS=30000,      # 30s connect timeout
        socketTimeoutMS=60000,       # 60s socket timeout  
        serverSelectionTimeoutMS=30000,  # 30s server selection
        heartbeatFrequencyMS=30000,  # 30s heartbeat
        retryWrites=True,            # Retry writes
        retryReads=True              # Retry reads
    )
    return client[db_name]
