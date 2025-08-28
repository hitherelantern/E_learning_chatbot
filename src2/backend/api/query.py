from fastapi import APIRouter, HTTPException, WebSocket
from backend.models.query_models import QueryRequest, QueryResponse, RetrievedDocument
from backend.core.milvus_utils import ask_query, ask_query_stream, generate_streaming_answer   # ✅ import your helper
from fastapi import Depends
from backend.db.db_manager import MongoDBManager2


db_manager_instance  = MongoDBManager2()
def get_db_manager():
    return db_manager_instance # Or return a global/singleton instance

router = APIRouter()

from langchain_core.documents import Document

@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest,db_manager: MongoDBManager2 = Depends(get_db_manager)):
    try:

        print(f"payload session id : {payload.session_id}")

        llm_answer, context = ask_query(
            c_name=payload.collection,
            query=payload.question,
            session_id=payload.session_id,
            db_manager= db_manager
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




@router.websocket("/ws/query")
async def websocket_query(ws: WebSocket,db_manager: MongoDBManager2 = Depends(get_db_manager)):
    await ws.accept()
    data = await ws.receive_json()
    c_name = data["collection"]
    query = data["question"]
    session_id = data["session_id"]

    async for chunk in ask_query_stream(c_name, query, session_id, db_manager):
        await ws.send_json(chunk)
        

    await ws.close()




from fastapi.responses import StreamingResponse

@router.post("/query/stream")
async def stream_query(payload: QueryRequest, db_manager: MongoDBManager2 = Depends(get_db_manager)):
    try:
        stream = await generate_streaming_answer(
            question=payload.question,
            collection=payload.collection,
            session_id=payload.session_id,
            db_manager=db_manager
        )
        return StreamingResponse(stream, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream error: {e}")

