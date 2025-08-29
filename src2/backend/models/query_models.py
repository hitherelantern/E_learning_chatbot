from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    collection: str
    question: str
    top_k: Optional[int] = None   # allow override
    session_id: str


class RetrievedDocument(BaseModel):
    text: str
    metadata: Optional[dict] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str] = None   # ✅ new
    results: List[RetrievedDocument]



# Define schema with Pydantic (for session wise summary)
class ConversationSummary(BaseModel):
    summary: str          # 2–3 sentence summary
    topics: List[str]     # 2–5 keywords