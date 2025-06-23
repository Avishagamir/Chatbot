import streamlit as st
from ChatbotEngineWithAI import AIBot

st.set_page_config(page_title="NYC Apartment Finder Bot", layout="wide")
st.title("🏙️ NYC Apartment Finder Bot")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = AIBot()

# Message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input at the bottom
user_input = st.chat_input("Type your message...")

# Process user input
if user_input:
    st.session_state.messages.append(("You", user_input))
    bot_reply = st.session_state.chatbot.process_message(user_input)
    st.session_state.messages.append(("Bot", bot_reply))

# Show chat history
for sender, message in st.session_state.messages:
    with st.chat_message(sender.lower()):
        st.markdown(message)
