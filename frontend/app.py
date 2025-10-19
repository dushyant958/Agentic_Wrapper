"""
Streamlit Frontend for Multi-Agent Research Assistant
"""
import streamlit as st
import requests
from typing import List, Dict
import time
import os
from dotenv import load_dotenv

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
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 4px solid #43A047;
    }
    .stButton>button {
        width: 100%;
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
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode",
        ["💬 Chat", "🔍 Research", "📄 Document Q&A", "🎤 Voice Transcription"],
        key="mode_selector"
    )
    
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
        key="model_selector"
    )
    
    st.markdown("---")
    
    # RAG Settings
    if mode == "💬 Chat":
        st.markdown("### 📚 RAG Settings")
        st.session_state.rag_enabled = st.checkbox(
            "Use uploaded documents",
            value=st.session_state.rag_enabled,
            help="Enable to query uploaded documents"
        )
        
        if st.session_state.rag_enabled:
            st.info(f"📁 {len(st.session_state.uploaded_files)} documents loaded")
    
    st.markdown("---")
    
    # System Stats
    st.markdown("### 📊 System Status")
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            st.metric("Documents", stats['rag']['total_documents'])
            st.metric("Chat Sessions", stats['chat']['active_sessions'])
            st.success("✅ All systems operational")
        else:
            st.error("❌ Backend unavailable")
    except:
        st.error("❌ Cannot connect to backend")
    
    st.markdown("---")
    
    # Clear buttons
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
        st.rerun()
    
    if st.button("🗑️ Clear Documents", use_container_width=True):
        requests.post(f"{API_URL}/clear")
        st.session_state.uploaded_files = []
        st.session_state.rag_enabled = False
        st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 AI Research Assistant</div>', unsafe_allow_html=True)

# ===================== CHAT MODE =====================
if mode == "💬 Chat":
    st.markdown("## 💬 Chat with AI")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 You: {msg["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🤖 Assistant: {msg["content"]}</div>', 
                          unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", key="chat_input")
        col1, col2 = st.columns([3, 1])
        
        with col2:
            submit = st.form_submit_button("Send 📤", use_container_width=True)
        
        if submit and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Show spinner while processing
            with st.spinner("🤔 Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": user_input,
                            "session_id": st.session_state.session_id,
                            "model": selected_model,
                            "use_rag": st.session_state.rag_enabled
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        assistant_message = result["response"]
                        
                        # Add assistant message to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_message
                        })
                        
                        # Show context if RAG was used
                        if result.get("rag_used") and result.get("context"):
                            with st.expander("📚 Context Used"):
                                for i, ctx in enumerate(result["context"], 1):
                                    st.markdown(f"**Context {i}:**\n{ctx[:300]}...")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")
            
            st.rerun()

# ===================== RESEARCH MODE =====================
elif mode == "🔍 Research":
    st.markdown("## 🔍 Multi-Agent Research")
    st.info("📝 Enter a research question and let our AI agents conduct comprehensive research for you!")
    
    with st.form("research_form"):
        research_query = st.text_area(
            "Research Question",
            placeholder="e.g., What are the latest developments in AI safety and alignment?",
            height=100
        )
        
        submit_research = st.form_submit_button("🚀 Start Research", use_container_width=True)
        
        if submit_research and research_query:
            with st.spinner("🔬 Research agents are working... This may take a few minutes..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Update progress
                    status_text.text("🔍 Planning research strategy...")
                    progress_bar.progress(20)
                    time.sleep(0.5)
                    
                    status_text.text("📚 Gathering information from multiple sources...")
                    progress_bar.progress(40)
                    
                    # Call research endpoint
                    response = requests.post(
                        f"{API_URL}/research",
                        json={
                            "query": research_query,
                            "model": selected_model
                        },
                        timeout=300  # 5 minutes timeout
                    )
                    
                    progress_bar.progress(70)
                    status_text.text("✅ Verifying information...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(90)
                    status_text.text("📝 Writing final report...")
                    
                    if response.status_code == 200:
                        result = response.json()
                        progress_bar.progress(100)
                        status_text.text("✅ Research completed!")
                        
                        st.success("Research completed successfully!")
                        
                        # Display results
                        st.markdown("### 📊 Research Results")
                        st.markdown(result["result"])
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Report",
                            data=result["result"],
                            file_name="research_report.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Research failed')}")
                        
                except requests.Timeout:
                    st.error("⏰ Research timed out. Please try a simpler query.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ===================== DOCUMENT Q&A MODE =====================
elif mode == "📄 Document Q&A":
    st.markdown("## 📄 Document Q&A")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, Images, or Text files",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
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
                                st.success(f"✅ {file.name} uploaded successfully!")
                            else:
                                st.error(f"❌ Failed to upload {file.name}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.markdown("### 📁 Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                st.markdown(f"- **{file_info['name']}** ({file_info['text_length']} chars)")
    
    with col2:
        st.markdown("### ❓ Ask Questions")
        
        if not st.session_state.uploaded_files:
            st.info("👈 Please upload documents first")
        else:
            with st.form("query_form"):
                query = st.text_input("Your question about the documents:")
                k_results = st.slider("Number of relevant chunks", 1, 10, 3)
                
                submit_query = st.form_submit_button("🔍 Search", use_container_width=True)
                
                if submit_query and query:
                    with st.spinner("Searching documents..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/query",
                                json={"query": query, "k": k_results}
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                st.markdown("### 💡 Answer")
                                st.success(result["answer"])
                                
                                with st.expander("📚 Relevant Context"):
                                    for i, ctx in enumerate(result["context"], 1):
                                        st.markdown(f"**Context {i}:**")
                                        st.text(ctx)
                                        st.markdown("---")
                            else:
                                st.error(f"Error: {response.json().get('detail')}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

# ===================== VOICE TRANSCRIPTION MODE =====================
elif mode == "🎤 Voice Transcription":
    st.markdown("## 🎤 Voice Transcription")
    st.info("Upload an audio file to transcribe it to text using Whisper AI")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Audio")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "webm", "ogg"],
            key="audio_uploader"
        )
        
        if audio_file:
            st.audio(audio_file, format=audio_file.type)
            
            if st.button("🎯 Transcribe", use_container_width=True):
                with st.spinner("🎤 Transcribing audio..."):
                    try:
                        files = {"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        response = requests.post(f"{API_URL}/transcribe", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("✅ Transcription completed!")
                            
                            # Display results
                            st.markdown("### 📝 Transcription")
                            st.text_area("Text", result["text"], height=200)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Language", result["language"].upper())
                            with col_b:
                                if result.get("duration"):
                                    st.metric("Duration", f"{result['duration']:.1f}s")
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Transcript",
                                data=result["text"],
                                file_name="transcript.txt",
                                mime="text/plain"
                            )
                            
                            # Option to use in chat
                            if st.button("💬 Use in Chat", use_container_width=True):
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": result["text"]
                                })
                                st.info("Transcript added to chat! Switch to Chat mode to continue.")
                        else:
                            st.error(f"Error: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("### ℹ️ Supported Formats")
        st.markdown("""
        - **MP3** - Most common format
        - **WAV** - Uncompressed audio
        - **M4A** - Apple audio format
        - **WEBM** - Web audio format
        - **OGG** - Open-source format
        
        **Tips:**
        - Clear audio = better transcription
        - Supports multiple languages
        - Max file size: 25MB
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🤖 Powered by Groq, CrewAI, and Streamlit</p>
</div>
""", unsafe_allow_html=True)