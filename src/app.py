import streamlit as st
from interface_helpers import *

# --- Streamlit App ---
st.title("E-Learning Assistant")
st.header("Query Transcriptions with Generative AI")

# Load existing DB collections
existing_collections = get_db_collections()

# --- Sidebar: Upload Audio ---
uploaded_audio = st.sidebar.file_uploader(
    "Upload an audio file", 
    type=["mp3", "wav", "webm", "m4a", "ogg","mp4"]
)

if uploaded_audio:
    # Sidebar: one-time collection name input
    c_name = st.sidebar.text_input("Enter a name for the collection")
    process_audio_upload(uploaded_audio,c_name)




# --- Sidebar: YouTube Link ---
youtube_url = st.sidebar.text_input("Or enter a YouTube URL")

if youtube_url and st.sidebar.button("Process YouTube Video"):
    with st.spinner("Downloading and processing YouTube audio..."):
        audio_path = download_audio_without_ffmpeg(youtube_url)
        st.session_state["yt_audio_path"] = audio_path
        st.session_state["show_c_name_input"] = True  # flag to show text box
    st.sidebar.success("YouTube video processed successfully!")

# Show collection name input ONLY if download finished
if st.session_state.get("show_c_name_input", False):
    c_name = st.sidebar.text_input("Enter a name for the collection", key="yt_c_name")

    if c_name:
        process_audio_upload(st.session_state["yt_audio_path"], c_name)
        st.sidebar.success("Audio added to database!")
        # cleanup so it doesn’t rerun every refresh
        st.session_state.pop("yt_audio_path", None)
        st.session_state["show_c_name_input"] = False





# --- Maintain state ---
selected_collection = st.session_state.get("selected_collection")

# --- Collection Dropdown ---
selected_collection = st.selectbox(
    "Query from an existing database",
    existing_collections,
    index=(
        existing_collections.index(selected_collection)
        if selected_collection in existing_collections
        else 0
    ),
    key="selected_collection"
)

# --- Query Input & Execution ---
query = st.text_input("Enter your question:")

if st.button("Execute Query") and query:
    ask_query(selected_collection, query)
    st.success(f"Query executed successfully for collection: {selected_collection}")

    if st.button("Reset"):
        st.session_state.selected_collection = None
        st.session_state.query_executed = False
