# create_collection, insert_chunks, search

import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection,db
)
from backend.core.config import load_config
from langchain_community.vectorstores import Milvus
from langchain.prompts import load_prompt
from backend.utils import load_file, resource_path_prompts, safe_run
from backend.db.db_manager import MongoDBManager2
from backend.core.memory import enhance_query





cfg = load_config()

emb_model = cfg.llm["embedding_model"]

llm = cfg.llm["chat_model"]["name"]
temperature = cfg.llm["chat_model"]["temperature"]



# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model=emb_model["name"])

# llm 
llm_stage1 = ChatGoogleGenerativeAI(model=llm, temperature=temperature)
llm_stage2 = ChatGoogleGenerativeAI(model=llm, 
                             temperature=temperature,
                             callbacks=[StreamingStdOutCallbackHandler()]  # prints tokens as they stream
                                )

connections.connect(
    host=cfg.milvus["host"],
    port=cfg.milvus["port"]
)
db.using_database(db_name=cfg.milvus["database"])


prompts = load_file(resource_path_prompts(cfg.retrieval["prompt_template"]))









    
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
    


def insert_chunks(chunks: list[str], collection_name: str, embedding_model, dim: int = cfg.collection["dim"]) -> None:
    if collection_name not in utility.list_collections():
        collection = create_collection(collection_name, dim=dim)

        index_params = cfg.collection["index_params"]

        if not any(index.field_name == "vector" for index in collection.indexes):
            collection.create_index(field_name="vector", index_params=index_params)
            print("Index created successfully.")
    else:
        collection = Collection(name=collection_name)

    # vectors = embedding_model.embed_documents(chunks)
    vectors = embed_in_batches(embedding_model,chunks)

    # Align with schema: [id, vector, text]
    data = [vectors, chunks]

    collection.insert(data)
    collection.load()
    print(f"Inserted {len(chunks)} records successfully into '{collection_name}'.")





def embed_in_batches(embedding_model, chunks, batch_size=emb_model["batch_size"], delay=emb_model["delay"]):
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"batch is {batch}")
        vectors.extend(embedding_model.embed_documents(batch))
        time.sleep(delay)  # avoid hitting per-minute limit
    return vectors




def search(query: str, top_k:int, collection_name: str):
    try:
        search_params = cfg.retrieval["search_params"]

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
    





@safe_run()
def ask_query(c_name: str, query: str, session_id: str, db_manager: MongoDBManager2):

    if not query:
        return None, []

    # Step 1: Enhance query
    enhanced_query = enhance_query(query, session_id, db_manager)
    print(f"Enhanced Query: {enhanced_query}")

    # Step 2: Search in Milvus
    results = search(enhanced_query, top_k=cfg.retrieval["top_k"], collection_name=c_name)
    retrieved = [result.page_content for result in results]
    transcription_context = "\n".join(retrieved)

    # Step 3: Build final answer prompt
    context_template = prompts["context_answer_prompt"]["template"]
    final_prompt = context_template.format(
        transcription_context=transcription_context,
        question=query
    )
    response = llm_stage1.invoke(final_prompt)

    # Step 4: Save conversation in Mongo
    # db_manager.save_message(session_id, query, response.content)

    return response.content, results




async def ask_query_stream(c_name: str, query: str, session_id: str, db_manager: MongoDBManager2):
    if not query:
        return

    # Step 1: Enhance query
    enhanced_query = enhance_query(query, session_id, db_manager)

    # Step 2: Search documents in Milvus
    results = search(
        enhanced_query, 
        top_k=cfg.retrieval["top_k"], 
        collection_name=c_name
    )
    transcription_context = "\n".join([r.page_content for r in results])

    # Step 3: Build prompt for LLM
    final_prompt = prompts["context_answer_prompt"]["template"].format(
        transcription_context=transcription_context,
        question=query
    )

    # Step 4: Stream response tokens
    full_answer = ""
    async for chunk in llm_stage2.astream(final_prompt):
        # Each chunk is an AIMessageChunk, usually with .content
        token = getattr(chunk, "content", None)
        if token:
            full_answer += token  # accumulate
            yield token
        else:
            yield {"debug": chunk.__dict__}  # fallback if needed

    # ✅ Prepare final payload
    final_payload = {
        "type": "final",
        "answer": full_answer,
        "retrieved_docs": [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None)
            } for doc in results
        ]
    }

    # ✅ Save into MongoDB before sending final message
    db_manager.save_message(session_id, query, full_answer)

 

async def generate_streaming_answer(
    question: str,
    collection: str,
    session_id: str,
    db_manager: MongoDBManager2
):
    # Step 1: Enhance query
    enhanced_query = enhance_query(question, session_id, db_manager)
    print(f"enhanced query:{enhanced_query}")
    results = search(
        enhanced_query,
        top_k=cfg.retrieval["top_k"],
        collection_name=collection
    )
    transcription_context = "\n".join([r.page_content for r in results])

    # Step 2: Build final prompt
    final_prompt = prompts["context_answer_prompt"]["template"].format(
        transcription_context=transcription_context,
        question=question
    )

    # Step 3: Streaming generator with DB save
    async def event_stream():
        full_answer = ""
        async for chunk in llm_stage1.astream(final_prompt):
            if chunk.content:
                full_answer += chunk.content
                yield chunk.content  # stream token to client

        # Save full answer to DB once streaming is done
        db_manager.save_message(session_id, question, full_answer)

    return event_stream()  # return the generator itself



