# FastAPI main app (endpoints)

from fastapi import FastAPI
from backend.api import collections, ingest, query

app = FastAPI(title="RAG Backend")

# Register routers
app.include_router(collections.router)
app.include_router(ingest.router)
app.include_router(query.router)



@app.get("/")
def root():
    return {"message": "Backend is running! Use /collections endpoints."}

