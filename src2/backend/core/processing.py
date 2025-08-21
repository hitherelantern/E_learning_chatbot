# clean_transcript, chunk_transcript

import re,spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import load_config


cfg = load_config()

nlp_model = spacy.load("en_core_web_sm")


def clean_transcript(transcript: str) -> str:
    transcript = re.sub(r"\b(um|ah|uh)\b", " ", transcript)
    transcript = re.sub(r"[^\w\s.,!?']", " ", transcript)
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript.lower()



def standardize_text(text: str):
    return text.lower()




def chunk_transcript(transcript: str):

    method = cfg.chunking["method"]

    if method == "recursive":
        params = cfg.chunking["recursive"]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["overlap"]
        )
        return splitter.split_text(transcript)

    elif method == "semantic":
        params = cfg.chunking["semantic"]
        return semantic_chunking_with_overlap(
            transcript,
            overlap=params["overlap"],
            window=params["window"]
        )



def semantic_chunking_with_overlap(text, overlap=1, window=5):
    doc = nlp_model(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    for i in range(0, len(sentences), window - overlap):
        chunk = " ".join(sentences[i:i + window])
        chunks.append(chunk)
    return chunks

