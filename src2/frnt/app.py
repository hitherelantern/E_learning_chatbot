# Streamlit main UI

import streamlit as st
from frnt.interface_helpers import create_collection, ingest_audio, ingest_youtube, query_collection

st.set_page_config(page_title="RAG System", layout="wide")

st.title("🎙️ Audio RAG with Milvus + Streamlit")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Collections", "Ingest", "Query"])



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
        # st.json(res)
        if res["status"] == "ok":
            st.write(f"Inserted {res["chunks"]} chunks successfully into collection {collection} !")

    # Option 2: YouTube
    st.subheader("YouTube URL")
    youtube_url = st.text_input("Paste YouTube URL")
    if youtube_url and st.button("Ingest from YouTube"):
        res = ingest_youtube(youtube_url, collection)
        # st.json(res)
        if res["status"] == "ok":
            st.write(f"Inserted {res["chunks"]} chunks successfully into collection {collection} !")



elif page == "Query":
    st.header("🔍 Query Collection")
    collection = st.text_input("Collection name")
    query_text = st.text_area("Enter your query")
    top_k = st.slider("Top K", 1, 10, 5)

    if st.button("Search"):
        if collection and query_text:
            res = query_collection(query_text, collection)

            # ✅ Just show the answer
            st.subheader("💡 Answer")
            st.write(res.get("answer", "No answer returned."))

        else:
            st.warning("Please fill in all fields.")

