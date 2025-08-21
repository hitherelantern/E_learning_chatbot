import os

# Define the folder structure
folders = [
    "src2/backend/api",
    "src2/backend/core",
    "src2/backend/models",
    "src2/frontend",
]

files = {
    "src2/backend/__init__.py": "",
    "src2/backend/server.py": "# FastAPI main app (endpoints)\n",
    "src2/backend/api/__init__.py": "",
    "src2/backend/api/collections.py": "# Endpoints for /collections\n",
    "src2/backend/api/ingest.py": "# Endpoints for /ingest/audio + /ingest/youtube\n",
    "src2/backend/api/query.py": "# Endpoints for /query\n",
    "src2/backend/core/__init__.py": "",
    "src2/backend/core/config.py": "# load_config()\n",
    "src2/backend/core/milvus_utils.py": "# create_collection, insert_chunks, search\n",
    "src2/backend/core/transcription.py": "# whisper transcription\n",
    "src2/backend/core/processing.py": "# clean_transcript, chunk_transcript\n",
    "src2/backend/models/__init__.py": "",
    "src2/backend/models/query_models.py": "# Pydantic models (e.g., QueryRequest)\n",
    "src2/frontend/__init__.py": "",
    "src2/frontend/app.py": "# Streamlit main UI\n",
    "src2/frontend/interface_helpers.py": "# Functions calling backend APIs\n",
    "src2/config.yaml": "# Shared configuration\n",
    "src2/.env": "# Environment variables\n",
    "src2/requirements.txt": "# Add Python dependencies here\n",
    "src2/README.md": "# Project Documentation\n",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Folder structure created in src2/")
