import pymongo
from datetime import datetime
from backend.core.config import load_config

cfg = load_config()

class MongoDBManager:
    def __init__(self, uri= cfg.database["server"], db_name=cfg.database["db"], collection_name=cfg.database["collection"]):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_chat(self, session_id, user_query, bot_answer, collection_name,context,top_k = cfg.retrieval["top_k"]):
        doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "collection": collection_name,
            "user_query": user_query,
            "bot_answer": bot_answer,
            "top_k": top_k,
            "context": context
        }
        return self.collection.insert_one(doc)

    def get_chats_by_session(self, session_id):
        return list(self.collection.find({"session_id": session_id}).sort("timestamp", 1))
