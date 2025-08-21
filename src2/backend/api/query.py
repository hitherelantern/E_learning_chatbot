from fastapi import APIRouter, HTTPException
from backend.models.query_models import QueryRequest, QueryResponse, RetrievedDocument
from backend.core.milvus_utils import ask_query   # ✅ import your helper

router = APIRouter()

from langchain_core.documents import Document

@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest):
    try:
        llm_answer, context = ask_query(
            c_name=payload.collection,
            query=payload.question
        )

        # 🔹 Convert LangChain Documents → RetrievedDocument
        retrieved_docs = [
            RetrievedDocument(
                text=doc.page_content,
                metadata=doc.metadata,
                score=getattr(doc, "score", None)  # safe fallback
            )
            for doc in context
        ]

        return QueryResponse(
            question=payload.question,
            answer=llm_answer,
            results=retrieved_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")
