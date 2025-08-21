# Functions calling backend APIs


import requests

BACKEND_URL = "http://localhost:8001"  # or your deployed FastAPI URL

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

def query_collection(query: str, collection: str, k: int = 5):
    res = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": query,       # ✅ renamed
            "collection": collection,
            "top_k": k              # ✅ renamed
        }
    )
    print("Status:", res.status_code)
    print("Text:", res.text)
    return res.json()




