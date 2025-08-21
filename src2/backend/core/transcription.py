# whisper transcription

import whisper
from backend.core.config import load_config
from backend.core import processing, milvus_utils
from backend.utils import safe_run


cfg = load_config()


def transcribe_audio(file_path):
    model = whisper.load_model(cfg.transcription["model_size"])
    result = model.transcribe(file_path)
    return result["text"]



@safe_run()
def process_audio_file(file_path: str, collection: str):
    """
    Shared ingestion pipeline (used by both API and Streamlit).
    """
    # 1. Transcribe
    transcript = transcribe_audio(file_path)

    # 2. Clean + Chunk
    cleaned = processing.clean_transcript(transcript)
    chunks = processing.chunk_transcript(cleaned)

    # 3. Insert into Milvus
    milvus_utils.insert_chunks(chunks, collection, milvus_utils.embedding_model)

    return {
        "status": "ok",
        "chunks": len(chunks),
        "transcript": transcript,
    }


def download_audio_without_ffmpeg(youtube_url, output_folder=cfg.transcription['audio_folder']):
    import os
    from yt_dlp import YoutubeDL
    
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        file_path = ydl.prepare_filename(info)  # final downloaded path
        print(f"file_path : {file_path}")
        return file_path
