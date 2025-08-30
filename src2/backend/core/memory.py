import os
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.db.db_manager import MongoDBManager2
from backend.core.config import load_config
from backend.utils import load_file, resource_path_prompts
from backend.models.query_models import ConversationSummary
from langchain.output_parsers import PydanticOutputParser


cfg = load_config()


llm = cfg.llm["chat_model"]["name"]
temperature = cfg.llm["chat_model"]["temperature"]
prompts = load_file(resource_path_prompts(cfg.retrieval["prompt_template"]))

llm = ChatGoogleGenerativeAI(model=llm, temperature=temperature)






"""
Note....

⚡ Difference:

If you use ChatPromptTemplate from LangChain → you keep structured messages (system, human, ai).

If you just load string templates from JSON like above → you directly .format() and feed the string into the model."""




summarize_prompt = ChatPromptTemplate.from_template(prompts["summarize_prompt"]["template"])
summarize_template = prompts["summarize_prompt"]["template"]





def enhance_query(query: str, session_id: str, db_manager: MongoDBManager2) -> str:

    """Enhance user query using past conversation from MongoDB."""
    # Get full chat history from Mongo
    messages = db_manager.get_chat_history(session_id)
    
    # Prepare last few exchanges for context
    history_text = "\n".join([
        f"User: {msg.get('user_query', '')}\nBot: {msg.get('bot_answer', '')}"
        for msg in messages[-5:]
    ])
    
    # Load enhancement prompt from JSON
    enhancement_template = prompts["enhancement_prompt"]["template"]
    enhancement_prompt = enhancement_template.format(
        chat_history=history_text,
        query=query
    )
    
    # Call LLM to rewrite the query
    enhanced_query = llm.invoke(enhancement_prompt).content.strip()
    return enhanced_query





def summarize_session(db_manager, user_id: str = None):
    """
    Summarize all chat sessions in MongoDB using the summarization LLM prompt.

    Args:
        db_manager: MongoDBManager2 instance
        user_id (str): ID of the user, defaults to system login name
    """

    # Create parser
    parser = PydanticOutputParser(pydantic_object=ConversationSummary)

    if user_id is None:
        user_id = os.getlogin()

    summarize_template = prompts["summarize_prompt"]["template"]
    format_instructions = parser.get_format_instructions()

    conversations = db_manager.get_all_conversations()

    for conv in conversations:
        session_id = conv["session_id"]

        # Get last 5 messages from session
        messages = db_manager.get_chat_history(session_id)
        history_text = "\n".join([
            f"User: {msg.get('user_query', '')}\nBot: {msg.get('bot_answer', '')}"
            for msg in messages[-5:]
        ])

        # Build final prompt
        final_prompt = summarize_template.format(
            chat_history=history_text,
            format_instructions=format_instructions
        )

        # Call LLM
        raw_summary = llm.invoke(final_prompt)

        try:
            parsed_summary = parser.parse(raw_summary.content)
        except Exception as e:
            print(f"❌ Failed to parse summary for {session_id}: {e}")
            continue

        # Save structured summary back into DB
        db_manager.save_summary(
            session_id=session_id,
            user_id=user_id,
            summary=parsed_summary.model_dump()  # store as JSON
        )

        print(f"✅ Finished summarizing {session_id}")
        print(f"   Summary: {parsed_summary.summary}")
        print(f"   Topics: {parsed_summary.topics}")





def enhance_query_wss(query: str, session_id: str, db_manager: MongoDBManager2) -> str:

    """Enhance user query using both recent history and wss (within session summary)"""
    
    # --- Get last few turns ---
    messages = db_manager.get_chat_history(session_id)
    history_text = "\n".join([
        f"User: {msg.get('user_query', '')}\nBot: {msg.get('bot_answer', '')}"
        for msg in messages[-5:]
    ])
    
    # --- Get summary (if exists) ---
    summary = db_manager.get_summary(session_id)


    session_summary = "\n".join([
                f"Summary: {s.get('summary', '')}\nkeywords: {s.get('topics', '')}"
                for s in summary
            ])
    
    # --- Build enhancement prompt ---
    enhancement_template = prompts["enhancement_prompt_wss"]["template"]
    enhancement_prompt = enhancement_template.format(
        chat_history=history_text,
        summary=session_summary,
        query=query
    )
    
    # --- Call LLM ---
    enhanced_query = llm.invoke(enhancement_prompt).content.strip()
    return enhanced_query





def enhance_query_ass(query: str, session_id: str, db_manager: MongoDBManager2) -> str:

    """Enhance user query using both recent history and ass (Across Sessions Summary)"""
    
    # --- Get last few turns ---
    messages = db_manager.get_chat_history(session_id)
    history_text = "\n".join([
        f"User: {msg.get('user_query', '')}\nBot: {msg.get('bot_answer', '')}"
        for msg in messages[-5:]
    ])



    session_summaries = db_manager.get_all_summaries()

    total_summary = "\n".join([
                f"Summary: {s.get('summary', '')}\nkeywords: {s.get('topics', '')}"
                for s in session_summaries
            ])
                
    
    # --- Build enhancement prompt ---
    enhancement_template = prompts["enhancement_prompt_ass"]["template"]
    enhancement_prompt = enhancement_template.format(
        chat_history=history_text,
        summary=total_summary,
        query=query
    )
    
    # --- Call LLM ---
    enhanced_query = llm.invoke(enhancement_prompt).content
    return enhanced_query





def maybe_summarize_session(session_id, db_manager:MongoDBManager2):

    """When to summarize the session...idea to be thought of ! Not using for now!"""

    messages = db_manager.get_chat_history(session_id)
    
    # 1. End-of-session summarization
    if session_inactive_for(session_id, minutes=30): # type: ignore
        summary = summarize_session(db_manager)
        db_manager.clear_old_messages(session_id)  # optional
    
    # 2. Rolling summarization
    elif len(messages) > 50:  
        summary = llm.summarize(messages[:40])  # summarize oldest part
        db_manager.append_summary(session_id, summary)
        db_manager.trim_messages(session_id, keep_last=10)





# if __name__ == "__main__":
#     m = MongoDBManager2()


#     x = m.get_all_session_ids()
#     print(x)
    
#     x = m.get_all_summaries()
#     print([i['summary'] for i in x ])



    