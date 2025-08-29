# Functions calling backend APIs


import json
import requests
import websocket
import streamlit as st
from backend.core.config import load_config

cfg = load_config()


BACKEND_URL = cfg.backend["server"]  # or your deployed FastAPI URL

def create_collection(name: str, dim: int = 768):
    res = requests.post(f"{BACKEND_URL}/collections/{name}", params={"dim": dim})
    return res.json()


def ingest_audio(file, collection: str):
    files = {"file": file}
    res = requests.post(f"{BACKEND_URL}/ingest/audio?collection={collection}", files=files)
    return res.json()

def ingest_youtube(url: str, collection: str):
    res = requests.post(f"{BACKEND_URL}/ingest/youtube", params={"url": url, "collection": collection})
    return res.json()

def query_collection(query: str, collection: str,session_id:str,  k: int = 5):
    res = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": query,       # ✅ renamed
            "collection": collection,
            "top_k": k,              # ✅ renamed
            "session_id":session_id
        }
    )
    print("Status:", res.status_code)
    # print("Text:", res.text)
    return res.json()

def list_collections():
    res = requests.get(f"{BACKEND_URL}/collections")
    return res.json()







# Modify your function to be a generator (Websocket way!)
def stream_response_generator(query: str, collection: str, session_id: str):
    ws = websocket.create_connection("ws://localhost:8001/ws/query")
    try:
        ws.send(json.dumps({"collection": collection, "question": query, "session_id": session_id}))

        while True:
            token = ws.recv()
            if not token:
                break
            yield token  # Yield each token to the Streamlit app
    except websocket.WebSocketConnectionClosedException:
        # Handle closed connection gracefully
        pass
    finally:
        if 'ws' in locals() and ws.connected:
            ws.close()





def stream_query(query, collection, session_id):
    url = f"{BACKEND_URL}/query/stream"
    payload = {
        "question": query,
        "collection": collection,
        "session_id": session_id,
        "top_k": 5
    }

    # Stream response
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")
