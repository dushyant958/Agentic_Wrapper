"""
Streamlit Frontend for Multi-Agent Research Assistant
Unified Interface with Normal Chat and Agentic Research Modes
"""
import streamlit as st
import requests
from typing import List, Dict
import time
import os
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder

load_dotenv()

# Backend API URL
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #BBDEFB;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #E0E0E0;
        border-left: 4px solid #43A047;
    }
    .research-message {
        background-color: #FFE0B2;
        border-left: 4px solid #FF9800;
    }
    .stButton>button {
        width: 100%;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .normal-mode {
        background-color: #4CAF50;
        color: white;
    }
    .research-mode {
        background-color: #FF9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "mode" not in st.session_state:
    st.session_state.mode = "normal"
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Mode selection - ONLY 2 OPTIONS
    st.markdown("### 🎯 Mode Selection")
    mode = st.radio(
        "Choose Mode:",
        ["💬 Normal Chat", "🔬 Agentic Research"],
        key="mode_selector",
        help="Normal: Regular chat with RAG | Research: Multi-agent research workflow"
    )
    
    # Set mode in session state
    st.session_state.mode = "normal" if "Normal" in mode else "research"
    
    st.markdown("---")
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    available_models = {
        "llama-3.1-8b-instant": "Fast & Efficient",
        "llama-3.3-70b-versatile": "High Quality",
        "llama-3.1-70b-versatile": "Balanced",
        "mixtral-8x7b-32768": "Long Context"
    }
    
    selected_model = st.selectbox(
        "Choose LLM",
        list(available_models.keys()),
        format_func=lambda x: f"{x} - {available_models[x]}",
        index=0 if st.session_state.mode == "normal" else 1,
        key="model_selector"
    )
    
    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### 📚 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, Images, TXT)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
        key="doc_uploader",
        help="Upload documents to enable RAG in Normal Chat mode"
    )
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        response = requests.post(f"{API_URL}/upload", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.uploaded_files.append({
                                "name": file.name,
                                "size": len(file.getvalue()),
                                "text_length": result["text_length"]
                            })
                            st.success(f"✅ {file.name}")
                        else:
                            st.error(f"❌ Failed: {file.name}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Show uploaded files count
    if st.session_state.uploaded_files:
        st.info(f"📁 {len(st.session_state.uploaded_files)} documents loaded")
        with st.expander("View uploaded files"):
            for file_info in st.session_state.uploaded_files:
                st.markdown(f"- **{file_info['name']}**")
    
    st.markdown("---")
    
    # System Stats
    st.markdown("### 📊 System Status")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", stats['rag']['total_documents'])
            with col2:
                st.metric("Sessions", stats['chat']['active_sessions'])
            st.success("✅ Online")
        else:
            st.error("❌ Backend issue")
    except:
        st.error("❌ Cannot connect")
    
    st.markdown("---")
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Docs", use_container_width=True):
            requests.post(f"{API_URL}/clear")
            st.session_state.uploaded_files = []
            st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 AI Research Assistant</div>', unsafe_allow_html=True)

# Mode indicator
mode_badge_class = "normal-mode" if st.session_state.mode == "normal" else "research-mode"
mode_text = "Normal Chat Mode" if st.session_state.mode == "normal" else "Agentic Research Mode"
st.markdown(f'<span class="mode-badge {mode_badge_class}">{mode_text}</span>', unsafe_allow_html=True)

st.markdown("---")

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 <b>You:</b> {msg["content"]}</div>', 
                      unsafe_allow_html=True)
        else:
            # Different styling for research vs normal responses
            msg_class = "research-message" if msg.get("mode") == "research" else "assistant-message"
            mode_indicator = "🔬" if msg.get("mode") == "research" else "🤖"
            st.markdown(f'<div class="chat-message {msg_class}">{mode_indicator} <b>Assistant:</b> {msg["content"]}</div>', 
                      unsafe_allow_html=True)
            
            # Show context if available
            if msg.get("context"):
                with st.expander(f"📚 Context Used ({len(msg['context'])} sources)"):
                    for i, ctx in enumerate(msg["context"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                        st.markdown("---")

st.markdown("---")

# Input area
st.markdown("### 💬 Your Message")

# Create columns for mic button and text input
col1, col2 = st.columns([1, 11])

with col1:
    # Audio recorder button
    audio_bytes = audio_recorder(
        text="🎤",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x",
        key="audio_recorder"
    )

# Handle audio transcription
if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
    st.session_state.last_audio_bytes = audio_bytes
    with st.spinner("🎤 Transcribing audio..."):
        try:
            files = {"audio_file": ("recording.wav", audio_bytes, "audio/wav")}
            response = requests.post(f"{API_URL}/transcribe", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.transcribed_text = result["text"]
                st.success(f"✅ Transcribed: {result['text'][:50]}...")
            else:
                st.error("❌ Transcription failed")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Text input
user_input = st.text_area(
    "Type your message or use voice recording:",
    value=st.session_state.transcribed_text,
    height=100,
    key="message_input",
    placeholder="Ask anything or click the mic button to record..."
)

# Send button
col_send1, col_send2, col_send3 = st.columns([8, 2, 2])

with col_send2:
    if st.button("🗑️ Clear Input", use_container_width=True):
        st.session_state.transcribed_text = ""
        st.rerun()

with col_send3:
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

# Process message
if send_button and user_input.strip():
    # Clear transcribed text
    st.session_state.transcribed_text = ""
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Determine if RAG should be used (only in normal mode with documents)
    use_rag = st.session_state.mode == "normal" and len(st.session_state.uploaded_files) > 0
    
    # Show appropriate spinner based on mode
    spinner_text = "🔬 Research agents working..." if st.session_state.mode == "research" else "🤔 Thinking..."
    
    with st.spinner(spinner_text):
        try:
            if st.session_state.mode == "research":
                # Call research endpoint
                response = requests.post(
                    f"{API_URL}/research",
                    json={
                        "query": user_input,
                        "model": selected_model
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_message = result["result"]
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_message,
                        "mode": "research"
                    })
                else:
                    st.error(f"Research failed: {response.json().get('detail', 'Unknown error')}")
            
            else:
                # Normal chat mode
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": user_input,
                        "session_id": st.session_state.session_id,
                        "model": selected_model,
                        "use_rag": use_rag
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_message = result["response"]
                    
                    # Add to history with context if available
                    message_data = {
                        "role": "assistant",
                        "content": assistant_message,
                        "mode": "normal"
                    }
                    
                    if result.get("rag_used") and result.get("context"):
                        message_data["context"] = result["context"]
                    
                    st.session_state.chat_history.append(message_data)
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
        except requests.Timeout:
            st.error("⏰ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.rerun()

# Info boxes based on mode
if st.session_state.mode == "normal":
    if not st.session_state.uploaded_files:
        st.info("💡 **Tip:** Upload documents in the sidebar to enable RAG-powered responses!")
    else:
        st.success(f"✅ RAG enabled with {len(st.session_state.uploaded_files)} document(s)")
else:
    st.warning("🔬 **Research Mode Active:** Responses will use multi-agent research workflow (may take longer)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🤖 Powered by Groq, CrewAI, and Streamlit | Voice: Whisper AI</p>
</div>
""", unsafe_allow_html=True)