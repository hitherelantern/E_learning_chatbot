# Endpoints for /ingest/audio + /ingest/youtube

# src/backend/api/ingest.py
from fastapi import APIRouter, Query, UploadFile, File, HTTPException
import tempfile, os

from backend.core.transcription import *

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("/audio")
async def ingest_audio(file: UploadFile = File(...), collection: str = "default"):
    try:
        print("Received file:", file.filename)
        print("Collection:", collection)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = process_audio_file(tmp_path, collection)

        os.remove(tmp_path)
        return result

    except Exception as e:
        import traceback
        print("Error during audio ingestion:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")






@router.post("/youtube")
async def ingest_youtube(
    url: str = Query(..., description="YouTube video URL"),
    collection: str = Query("default", description="Target Milvus collection")
):
    try:
        # 1. Download audio (returns local file path)
        file_path = download_audio_without_ffmpeg(url)

        # 2. Process (transcribe -> clean -> chunk -> insert to Milvus)
        result = process_audio_file(file_path, collection)

        # 3. Cleanup (optional: keep cache if needed)
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
