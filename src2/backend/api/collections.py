# Endpoints for /collections


from fastapi import APIRouter, HTTPException
from backend.core import milvus_utils, config

router = APIRouter(prefix="/collections", tags=["collections"])

# Load config (host, port, etc.)
app_config = config.load_config()



@router.post("/{collection_name}")
def create_collection(collection_name: str, dim: int = 768):
    """
    Create a Milvus collection if it doesn't exist.
    """
    try:
        milvus_utils.create_collection(collection_name=collection_name, dim=dim)
        return {"status": "success", "message": f"Collection '{collection_name}' created."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/")
def list_collections():
    """
    List all available collections in Milvus.
    """
    from pymilvus import utility

    try:
        collections = utility.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.delete("/{collection_name}")
def drop_collection(collection_name: str):
    """
    Drop (delete) a collection from Milvus.
    """
    from pymilvus import utility

    try:
        if collection_name not in utility.list_collections():
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
        utility.drop_collection(collection_name)
        return {"status": "success", "message": f"Collection '{collection_name}' dropped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
