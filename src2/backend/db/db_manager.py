from zoneinfo import ZoneInfo
import pymongo
from datetime import datetime, timezone
from backend.core.config import load_config


cfg = load_config()

class MongoDBManager:
    def __init__(self, uri= cfg.database["server"], db_name=cfg.database["db"], collection_name=cfg.database["chat_history"]):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_chat(self, session_id, user_query, bot_answer, collection_name,context,top_k:int = 5):
        doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "collection": collection_name,
            "user_query": user_query,
            "bot_answer": bot_answer,
            "top_k": cfg.retrieval["top_k"],
            "context": context
        }
        return self.collection.insert_one(doc)

    def get_chats_by_session(self, session_id):
        return list(self.collection.find({"session_id": session_id}).sort("timestamp", 1))
    
    def get_all_sessions(self):
        return list(self.collection.aggregate([
            {"$group": {"_id": "$session_id"}}
        ]))




from pymongo import MongoClient
from datetime import datetime
from typing import Optional, List, Dict

class MongoDBManager2:
    def __init__(self, uri: str = cfg.database["server"], db_name: str = cfg.database["db"]):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.conversations = self.db[cfg.database["conversations"]]
        self.messages = self.db[cfg.database["chat_history"]]
        self.session_summary = self.db[cfg.database["session_summary"]]

    # ---------------------------
    # Conversation Methods
    # ---------------------------

    def create_conversation(self, session_id: str, first_message: Optional[str] = None) -> None:
        """Create a new conversation with dynamic name based on first message."""
        now = datetime.utcnow().isoformat()
        # Generate name
        if first_message:
            preview = " ".join(first_message.split()[:5])  # first 5 words
            name = f"Chat: {preview}"
        else:
            name = "New Chat"

        doc = {
            "session_id": session_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "archived": False,
        }
        # Use upsert so we don't duplicate if already exists
        self.conversations.update_one(
            {"session_id": session_id},
            {"$setOnInsert": doc},
            upsert=True
        )

    def get_conversation(self, session_id: str) -> Optional[Dict]:
        """Fetch a single conversation by session_id."""
        return self.conversations.find_one({"session_id": session_id, "archived": False})

    def update_conversation_timestamp(self, session_id: str) -> None:
        """Update 'updated_at' when new message is added."""
        self.conversations.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": datetime.utcnow().isoformat()}}
        )

    def get_all_conversations(self) -> List[Dict]:
        """Get all non-archived conversations sorted by recent update."""
        return list(self.conversations.find({"archived": False}).sort("updated_at", -1))

    def archive_conversation(self, session_id: str) -> None:
        """Soft delete conversation by marking it archived."""
        self.conversations.update_one(
            {"session_id": session_id},
            {"$set": {"archived": True}}
        )

    # ---------------------------
    # Message Methods
    # ---------------------------

    def save_message(self, session_id: str, user_query: str, bot_answer: str, top_k: int = 5, context: List = None) -> None:
        """Save a user-bot message pair to DB."""
        try:
            doc = {
                "session_id": session_id,
                "user_query": user_query,
                "bot_answer": bot_answer,
                # "context": context or [],
                "top_k": cfg.retrieval["top_k"],
                "timestamp": datetime.utcnow().isoformat()
            }

           
            self.messages.insert_one(doc)
 
            # Update conversation timestamp
            self.update_conversation_timestamp(session_id)
        

        except Exception as e:
            print(f"Exception is {e}")

            

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get all messages in a conversation sorted by time."""
        return list(self.messages.find({"session_id": session_id}).sort("timestamp", 1))
    

    def save_summary(self,session_id:str,user_id:str,summary:dict) -> None:

        try:
            doc = {
                "user_id":user_id,
                "session_id":session_id,
                "summary" : summary.get('summary',""),
                "topics": summary.get('topics',[]),
                "updated_at":datetime.now().astimezone().isoformat()
            }
            self.session_summary.insert_one(doc)


        except Exception as e:
            print(f"Error saving summary : {e}")
    

    def get_summary(self, session_id: str) -> List[Dict]:
        """Get summary of a particular session sorted by time."""
        return list(self.messages.find({"session_id": session_id}).sort("updated_at", 1))

    def get_all_session_ids(self) -> List[str]:
        """Gets all session ids from the corresponding collection"""
        return self.session_summary.distinct("session_id")
    
    def get_all_summaries(self) -> List[dict]:
        """Return all records in session_summary collection."""
        return list(self.session_summary.find({}).sort("updated_at", 1))



    # def generate_name_from_message(self, message: str) -> str:
    #     # Call an LLM here for better summarization
    #     prompt = f"Generate a short, descriptive title for this user query: {message}"
    #     response = llm(prompt)
    #     return response.content.strip()



