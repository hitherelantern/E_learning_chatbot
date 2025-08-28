import streamlit as st
from frnt.interface_helpers import create_collection, ingest_audio, ingest_youtube, query_collection
from backend.db.db_manager import MongoDBManager2
import uuid
from backend.api.query import db_manager_instance as db



# Generate session ID for user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.set_page_config(page_title="RAG System", layout="wide")
st.title("🎙️ Audio RAG with Milvus + Streamlit")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Collections", "Ingest", "Chat"])

if page == "Collections":
    st.header("📂 Manage Collections")
    col_name = st.text_input("Collection name")
    if st.button("Create Collection"):
        if col_name:
            res = create_collection(col_name)
            st.json(res)
        else:
            st.warning("Please enter a collection name.")

elif page == "Ingest":
    st.header("🎧 Ingest Data")
    collection = st.text_input("Collection to ingest into")

    # Option 1: Upload audio
    st.subheader("Upload Audio File")
    audio_file = st.file_uploader("Upload .mp3/.wav", type=["mp3", "wav", "webm", "m4a", "ogg","mp4"])
    if audio_file and st.button("Ingest Audio"):
        res = ingest_audio(audio_file, collection)
        if res["status"] == "ok":
            st.success(f"Inserted {res['chunks']} chunks successfully into collection {collection}!")

    # Option 2: YouTube
    st.subheader("YouTube URL")
    youtube_url = st.text_input("Paste YouTube URL")
    if youtube_url and st.button("Ingest from YouTube"):
        res = ingest_youtube(youtube_url, collection)
        if res["status"] == "ok":
            st.success(f"Inserted {res['chunks']} chunks successfully into collection {collection}!")
        else:
            st.write(f"some error occured!")

elif page == "Chat":
    from frnt.chat_ui import chat_interface
    chat_interface(db)
