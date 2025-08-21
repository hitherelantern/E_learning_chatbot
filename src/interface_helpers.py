import streamlit as st
import os, time, re, whisper
import spacy
import yaml
from dotenv import load_dotenv
from pymilvus import utility, db, connections
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import load_prompt

# Local imports
from ingestion import *
# ---------------- Load Config ---------------- #
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()



# ---------------- Env + Models ---------------- #
load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model=config["llm"]["embedding_model"])
llm = ChatGoogleGenerativeAI(model=config["llm"]["chat_model"])




# ---------------- Milvus Connection ---------------- #
if "milvus_connected" not in st.session_state:
    connections.connect(
        host=config["milvus"]["host"],
        port=config["milvus"]["port"]
    )
    db.using_database(db_name=config["milvus"]["database"])
    st.session_state.milvus_connected = True




# ---------------- Transcription ---------------- #
def transcribe_audio(file_path):
    model = whisper.load_model(config["transcription"]["model_size"])
    result = model.transcribe(file_path)
    return result["text"]

def process_audio_upload(uploaded_audio,c_name):

    if not c_name.strip():
        st.sidebar.error("Collection name cannot be empty!")
        print("Returned to main function")
        return
        

    if c_name in utility.list_collections():
        st.sidebar.warning(f"Collection '{c_name}' already exists.")
        return

    try:
        total_start_time = time.time()
        trans = config["transcription"]

        # Save uploaded audio temporarily
        if not isinstance(uploaded_audio, str):
            
            print(f"INPUT AUDIO IS NOT A STRING!")
            temp_audio_path = os.path.join(trans["audio_folder"], uploaded_audio.name)
            os.makedirs(trans["audio_folder"], exist_ok=True)

            # ✅ Check if file already exists
            if os.path.exists(temp_audio_path):
                print(f"File already exists at {temp_audio_path}, skipping save.")
            else:
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                print(f"Saved new file at {temp_audio_path}")


        else:
            print(f"INPUT IS A STRING")
            temp_audio_path = uploaded_audio
         

        # Transcribe audio
        st.sidebar.info("Transcribing audio...")
        transcription_start_time = time.time()
        transcription = transcribe_audio(temp_audio_path)
        transcription_end_time = time.time()
        os.remove(temp_audio_path)

        # Saving the transcript
        txt_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
        temp_txt_path = os.path.join(trans["transcriptions_folder"], txt_name + ".txt")
        os.makedirs(trans["transcriptions_folder"], exist_ok=True)
        with open(temp_txt_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        # Clean + chunk transcription
        st.sidebar.info("Processing transcription...")
        processing_start_time = time.time()
        cleaned_content = clean_transcript(transcription)
        chunks = chunk_transcript(cleaned_content)
        processing_end_time = time.time()

        # Insert into Milvus
        insertion_start_time = time.time()
        insert_chunks(chunks, c_name, embedding_model)
        insertion_end_time = time.time()

        # Metrics
        transcription_time = transcription_end_time - transcription_start_time
        processing_time = processing_end_time - processing_start_time
        insertion_time = insertion_end_time - insertion_start_time
        total_time = time.time() - total_start_time

        st.sidebar.success(f"Processed {len(chunks)} chunks into '{c_name}'.")
        st.sidebar.write(f"Transcription Time: {transcription_time:.2f} s")
        st.sidebar.write(f"Processing Time: {processing_time:.2f} s")
        st.sidebar.write(f"Insertion Time: {insertion_time:.2f} s")
        st.sidebar.write(f"Total Time: {total_time:.2f} s")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")







# ---------------- Query Function ---------------- #
def ask_query(c_name, query):
    if query:
        results = search(query, k=config["retrieval"]["top_k"], collection_name=c_name)
        transcription_context = "\n".join([result.page_content for result in results])

        # Load prompt
        system_prompt = load_prompt(config["retrieval"]["prompt_template"])
        prompt = system_prompt.invoke({
            "transcription_context": transcription_context,
            "question": query
        })
        response = llm.invoke(prompt)

        st.write("### Response:")
        st.write(response.content)






# ---------------- Collection Utils ---------------- #
def get_db_collections():
    try:
        return utility.list_collections()
    except Exception as e:
        st.error(f"Error fetching collections: {e}")
        return []



# ----------------download yt. video -> audio--------------#

def download_audio_without_ffmpeg(youtube_url, output_folder=config['transcription']['audio_folder']):
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
