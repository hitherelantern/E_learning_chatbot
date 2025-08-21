from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    collection: str
    question: str
    top_k: Optional[int] = None   # allow override


class RetrievedDocument(BaseModel):
    text: str
    metadata: Optional[dict] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str] = None   # ✅ new
    results: List[RetrievedDocument]
