import re
import spacy
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema,
    connections, utility, db
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus



# ---------------- Load Config ---------------- #
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


# ---------------- Environment + Models ---------------- #
load_dotenv()
nlp_model = spacy.load("en_core_web_sm")

embedding_model = GoogleGenerativeAIEmbeddings(
    model=config["llm"]["embedding_model"]
)



# ---------------- Milvus Connection ---------------- #
connections.connect(
    host=config["milvus"]["host"],
    port=config["milvus"]["port"]
)
db.using_database(db_name=config["milvus"]["database"])




# ---------------- Preprocessing ---------------- #
def clean_transcript(transcript: str) -> str:
    transcript = re.sub(r"\b(um|ah|uh)\b", " ", transcript)
    transcript = re.sub(r"[^\w\s.,!?']", " ", transcript)
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript.lower()



def standardize_text(text: str):
    return text.lower()




def chunk_transcript(transcript: str):

    method = config["chunking"]["method"]

    if method == "recursive":
        params = config["chunking"]["recursive"]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["overlap"]
        )
        return splitter.split_text(transcript)

    elif method == "semantic":
        params = config["chunking"]["semantic"]
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



def add_metadata(chunks, timestamps, source):
    pass  # Placeholder





# ---------------- Milvus Integration ---------------- #
def create_collection(collection_name: str, dim: int):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]

    schema = CollectionSchema(fields=fields, description=f"Schema for {collection_name}")

    if collection_name not in utility.list_collections():
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully.")
        return collection
    else:
        return Collection(name=collection_name)
    


def insert_chunks(chunks: list[str], collection_name: str, embedding_model, dim: int = 768) -> None:
    if collection_name not in utility.list_collections():
        collection = create_collection(collection_name, dim=dim)

        index_params = config["collection"]["index_params"]

        if not any(index.field_name == "vector" for index in collection.indexes):
            collection.create_index(field_name="vector", index_params=index_params)
            print("Index created successfully.")
    else:
        collection = Collection(name=collection_name)

    vectors = embedding_model.embed_documents(chunks)

    # Align with schema: [id, vector, text]
    data = [vectors, chunks]

    collection.insert(data)
    collection.load()
    print(f"Inserted {len(chunks)} records successfully into '{collection_name}'.")



def search(query: str, k:int, collection_name: str):
    try:
        search_params = config["retrieval"]["search_params"]
        top_k = config["retrieval"]["top_k"]

        retriever = Milvus(
            embedding_function=embedding_model,
            collection_name=collection_name,
            connection_args={"host": config["milvus"]["host"], "port": config["milvus"]["port"]},
            search_params=search_params
        ).as_retriever()

        results = retriever.get_relevant_documents(query, top_k=top_k)
        return results

    except Exception as e:
        raise RuntimeError(f"The exception from retrieval component: {e}")
