import streamlit as st
import uuid
from decision import classify_input
from greetings import generate_greeting
from recomendation import ask_ai_recomendation
from consultation import ask_ai_consultation
from general_Information import ask_ai_general

# Set page config
st.set_page_config(
    page_title="Cybot - Asisten Gizi",
    page_icon="🍎",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for user profile
with st.sidebar:
    st.title("Profil Pengguna")
    
    # User profile inputs
    name = st.text_input("Nama", "Raihan")
    gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    age = st.number_input("Usia", min_value=1, max_value=100, value=25)
    height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=77)
    activity = st.selectbox(
        "Tingkat Aktivitas",
        ["Sangat Jarang", "Jarang", "Normal", "Sering", "Sangat Sering"],
        index=2
    )

# Main chat interface
st.title("Cybot - Asisten Gizi 🍎")
st.markdown("Selamat datang! Saya adalah Cybot, asisten gizi pribadi Anda. Saya dapat membantu memberikan rekomendasi diet dan makanan berdasarkan profil Anda.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response based on input classification
    with st.chat_message("assistant"):
        classify_result = classify_input(prompt)
        
        if classify_result == "greetings":
            response = generate_greeting(prompt, name)
        elif classify_result in ["recommendation", "rekomendasi"]:
            response = ask_ai_recomendation(name, activity, weight, height, age, gender, prompt)
        elif classify_result in ["consultation", "konsultasi"]:
            response = ask_ai_consultation(name, activity, weight, height, age, gender, prompt)
        elif classify_result == "general":
            response = ask_ai_general(prompt)
        else:
            response = "Maaf, saya tidak mengerti pertanyaan Anda. Silakan coba tanyakan hal lain."
            
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a clear chat button
if st.button("Bersihkan Percakapan"):
    st.session_state.messages = []
    st.rerun()
