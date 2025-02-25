import streamlit as st
import requests
import json
import os

# Set page config
st.set_page_config(
    page_title="SofaBed.com Chatbot",
    page_icon="🛋️",
    layout="centered"
)

# Use local development URL instead of Heroku
BACKEND_URL = "https://sofabed-chatbot-backend.onrender.com"  # We'll get this URL after deploying to Render

# Initialize session state for chat history and user input
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# Add custom CSS
st.markdown("""
<style>
.chat-container {
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}
.user-message {
    background-color: #e6f3ff;
    text-align: right;
}
.bot-message {
    background-color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🛋️ SofaBed.com Assistant")
st.markdown("Ask me anything about our sofa beds and furniture!")

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-container user-message">
            <b>You:</b> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container bot-message">
            <b>Assistant:</b> {content}
        </div>
        """, unsafe_allow_html=True)

def process_input():
    if st.session_state.user_input and st.session_state.user_input != st.session_state.current_input:
        user_input = st.session_state.user_input
        st.session_state.current_input = user_input
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Send request to FastAPI backend
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"question": user_input})
            )
            
            if response.status_code == 200:
                bot_response = response.json()["answer"]
                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                # Clear the input
                st.session_state.user_input = ""
            else:
                st.error(f"Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend server. Make sure it's running!")

# Chat input with placeholder text
st.text_input(
    "Type your question here...", 
    key="user_input", 
    on_change=process_input,
    value=""  # This ensures the input is empty by default
)

# Add some helpful information at the bottom
with st.expander("ℹ️ How to use this chatbot"):
    st.markdown("""
    1. Type your question about our sofa beds, furniture, or services
    2. Press Enter to get an answer
    3. The chatbot will respond based on information from sofabed.com
    
    Example questions:
    - What types of sofa beds do you offer?
    - What are your delivery options?
    - Do you have leather sofa beds?
    """) 