# create_collection, insert_chunks, search

from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection,db
)
from backend.core.config import load_config
from langchain_community.vectorstores import Milvus
from langchain.prompts import load_prompt
from backend.utils import resource_path_prompts




cfg = load_config()

embedding_model = GoogleGenerativeAIEmbeddings(
    model=cfg.llm["embedding_model"]
)

llm = ChatGoogleGenerativeAI(model=cfg.llm["chat_model"])


connections.connect(
    host=cfg.milvus["host"],
    port=cfg.milvus["port"]
)
db.using_database(db_name=cfg.milvus["database"])



    
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

        index_params = cfg.collection["index_params"]

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



def search(query: str, top_k:int, collection_name: str):
    try:
        search_params = cfg.retrieval["search_params"]
        top_k = cfg.retrieval["top_k"]

        retriever = Milvus(
            embedding_function=embedding_model,
            collection_name=collection_name,
            connection_args={"host": cfg.milvus["host"], "port": cfg.milvus["port"]},
            search_params=search_params
        ).as_retriever()

        results = retriever.get_relevant_documents(query, top_k=top_k)
        return results

    except Exception as e:
        raise RuntimeError(f"The exception from retrieval component: {e}")
    


def ask_query(c_name, query):
    if query:
        results = search(query, top_k=cfg.retrieval["top_k"], collection_name=c_name)
        retrieved = [result.page_content for result in results]
        transcription_context = "\n".join(retrieved)

        # Load prompt
        template_file = resource_path_prompts(cfg.retrieval["prompt_template"])
        system_prompt = load_prompt(template_file)
        prompt = system_prompt.invoke({
            "transcription_context": transcription_context,
            "question": query
        })
        response = llm.invoke(prompt)
        return response.content,results
