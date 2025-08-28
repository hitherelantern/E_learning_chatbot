import streamlit as st
from datetime import datetime
from typing import Optional
from uuid import uuid4
from .interface_helpers import query_collection, list_collections, stream_query,stream_response_generator





def scroll_to_bottom():
    st.markdown(
        """
        <script>
        var chatContainer = window.parent.document.querySelector('.stChatMessageContainer');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """,
        unsafe_allow_html=True
    )



def _ensure_session():
    """Ensure a unique session_id exists in st.session_state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())


def _render_sidebar(db):
    """Sidebar with conversations and New Chat button."""
    st.sidebar.title("📜 Conversations")

    if st.sidebar.button("➕ New chat"):
        # Generate a new session_id and clear history
        st.session_state.session_id = str(uuid4())
        st.session_state.chat_history = []
        st.rerun()

    conversations = db.get_all_conversations()
    if conversations:
        st.sidebar.subheader("Previous Chats")
        for conv in conversations:
            label = conv.get("name", "Untitled")
            if st.sidebar.button(f"🗨 {label}", key=f"conv-{conv['session_id']}"):
                st.session_state.session_id = conv["session_id"]
                # Load chat history for that session
                st.session_state.chat_history = db.get_chat_history(conv["session_id"])
                st.rerun()


def update_collection():
    """Callback function to update the selected collection in session state."""
    st.session_state.selected_collection = st.session_state.collection_select



# Decorate the chat-specific function as an experimental fragment
@st.fragment
def _chat_fragment(db,prompt):
    
     scroll_to_bottom()

     if prompt and st.session_state.selected_collection:
        # Ensure conversation exists
        if not db.get_conversation(st.session_state.session_id):
            db.create_conversation(st.session_state.session_id, first_message=prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Call the generator and pass it directly to st.write_stream()
            # response_generator = stream_response_generator(
            #     prompt,
            #     st.session_state.selected_collection,
            #     st.session_state.session_id,
            # )
            response_generator = stream_query(
                prompt,
                st.session_state.selected_collection,
                st.session_state.session_id
            )
            st.write_stream(response_generator)




def chat_interface(db):
    st.title("💬 Chat with Your Data")
    _ensure_session()
    _render_sidebar(db)

    collections = list_collections()['collections']

    # Initialize or validate selected_collection
    if "selected_collection" not in st.session_state or st.session_state.selected_collection not in collections:
        if collections:
            st.session_state.selected_collection = collections[0]
        else:
            st.session_state.selected_collection = None

    index = collections.index(st.session_state.selected_collection) if st.session_state.selected_collection in collections else 0

    st.sidebar.selectbox(
        'Choose a collection:',
        collections,
        index=index,
        key="collection_select",
        on_change=update_collection
    )

    # Load and render history (but avoid re-streaming)
    history = db.get_chat_history(st.session_state.session_id)
    for chat in history:
        with st.chat_message("user"):
            st.markdown(chat["user_query"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot_answer"])

    # Chat input
    prompt = st.chat_input("Type your question...")

   # Call the chat fragment to render the chat-specific parts
    _chat_fragment(db,prompt)

        # # Save final answer to DB
        # db.save_message(
        #     session_id=st.session_state.session_id,
        #     user_query=prompt,
        #     bot_answer=final_answer,
        #     context=[]
        # )

    
