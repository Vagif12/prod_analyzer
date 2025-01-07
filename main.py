import streamlit as st
from services.ai_assistant_service import AssistantService
from services.ai_prompts_service import PromptType
import base64



st.title("Production Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_history_summary = ""
    # st.session_state.first_response_called = False
    st.session_state.discussion_finished = False
    st.session_state.messages_called = 0

running_level, running_reason, dedication_level, user_input,  = None, None, None, None

assistant_svc = AssistantService(st.session_state.messages)

uploaded_file = st.file_uploader("Choose a file...")
if uploaded_file is not None:

    file_bytes = uploaded_file.read()
    # Determine the file format from the uploaded file name
    file_format = uploaded_file.name.split('.')[-1]  # Example: 'jpg', 'png', etc.
    
    # Encode the file bytes to base64
    image_64b = base64.b64encode(file_bytes).decode('utf-8')

    decoded_content = assistant_svc.decode_image(image_64b,file_format)
    
    with st.chat_message("assistant"):
        box = st.empty()
        result = assistant_svc.send_input(decoded_content, box, PromptType.PRODUCTION_PROMPT)
        st.session_state.messages.append({"role": "user", "content": decoded_content})
        st.session_state.messages.append({"role": "assistant", "content": result})

user_input = st.chat_input("Or type in a question...")
if user_input:   
    with st.chat_message("assistant"):
        box = st.empty()
        result = assistant_svc.send_input(user_input, box, PromptType.PRODUCTION_PROMPT)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": result})
        # st.session_state.first_response_called = True

