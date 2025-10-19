"""
FastAPI Backend for Multi-Agent Research Assistant
Complete with RAG, Chat, Research, and Transcription
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Import handlers
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_Ingestion.pdf_handler import PDFHandler
from Data_Ingestion.image_handler import ImageHandler
from models.embedder_model import EmbedderModel
from models.ocr_model import OCRModel
from Embedder.embedder import RAGPipeline
from Orchestrator.crew import ResearchCrew

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Research Assistant API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables!")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✓ Groq client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}")
        groq_client = None

# Initialize handlers
pdf_handler = PDFHandler()
image_handler = ImageHandler(languages=['en'])
ocr_model = OCRModel(languages=['en'], gpu=False)
embedder_model = EmbedderModel()
rag_pipeline = RAGPipeline(embedder_model)

# Initialize Research Crew (will be set on startup)
research_crew = None

# In-memory chat history (will be replaced with DB later)
chat_sessions: Dict[str, List[Dict]] = {}

# Available models
AVAILABLE_MODELS = {
    "llama-3.1-8b-instant": "Fast and efficient for quick responses",
    "llama-3.3-70b-versatile": "High quality, detailed responses",
    "llama-3.1-70b-versatile": "Balanced performance",
    "mixtral-8x7b-32768": "Great for long context"
}


@app.on_event("startup")
async def startup_event():
    """Initialize all models and systems on server startup"""
    global research_crew
    
    try:
        logger.info("=" * 60)
        logger.info("Starting Research Assistant API")
        logger.info("=" * 60)
        
        # Check if Groq is available
        if not groq_client:
            logger.warning("⚠️  Groq client not available - chat and transcription features will be disabled")
        else:
            logger.info("✓ Groq client ready")
        
        # Initialize PDF handler
        logger.info("✓ PDF handler initialized")
        
        # Initialize OCR
        if ocr_model.is_available():
            logger.info("✓ OCR model ready")
        else:
            logger.warning("⚠️  OCR model not available")
        
        # Initialize embedder
        if embedder_model.is_available():
            logger.info("✓ Embedding model ready")
        else:
            logger.warning("⚠️  Embedding model not available")
        
        # Initialize RAG pipeline
        logger.info("✓ RAG pipeline initialized")
        
        # Initialize Research Crew
        try:
            research_crew = ResearchCrew(llm_choice="llama-3.3-70b-versatile")
            logger.info("✓ Research Crew initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Research Crew: {str(e)}")
            logger.warning("⚠️  Research features will be disabled")
        
        logger.info("=" * 60)
        logger.info("All systems initialized successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str = "llama-3.1-8b-instant"
    use_rag: bool = False

class ChatResponse(BaseModel):
    response: str
    model_used: str
    session_id: str
    rag_used: bool = False
    context: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    sources: Optional[List[str]] = None

class UploadResponse(BaseModel):
    message: str
    filename: str
    text_length: int
    success: bool

class ResearchRequest(BaseModel):
    query: str
    model: str = "llama-3.3-70b-versatile"

class ResearchResponse(BaseModel):
    result: str
    status: str
    message: str

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: Optional[float] = None


# ===================== ENDPOINTS =====================

@app.get("/")
async def root():
    """Health check and system status"""
    return {
        "status": "healthy",
        "message": "Research Assistant API is running",
        "version": "1.0.0",
        "features": {
            "chat": groq_client is not None,
            "rag": True,
            "research": research_crew is not None,
            "transcription": groq_client is not None,
            "file_upload": True
        },
        "models": {
            "groq_available": groq_client is not None,
            "ocr_available": ocr_model.is_available(),
            "embedder_available": embedder_model.is_available(),
            "research_crew_available": research_crew is not None,
            "available_llms": list(AVAILABLE_MODELS.keys())
        },
        "rag_stats": rag_pipeline.vector_store.get_stats()
    }


# ===================== CHAT ENDPOINT =====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Simple chat with LLM, optionally using RAG context
    
    Args:
        message: User message
        session_id: Chat session identifier
        model: LLM model to use
        use_rag: Whether to use RAG context from uploaded documents
    """
    try:
        if not groq_client:
            raise HTTPException(
                status_code=503,
                detail="Groq API not available. Please check GROQ_API_KEY environment variable."
            )
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Initialize session if doesn't exist
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
        
        # Get RAG context if requested
        rag_context = None
        context_list = []
        if request.use_rag:
            stats = rag_pipeline.vector_store.get_stats()
            if stats['total_documents'] > 0:
                results = rag_pipeline.query(request.message, k=3)
                if results:
                    context_list = [text for text, _ in results]
                    rag_context = "\n\n".join(context_list)
        
        # Build messages
        messages = []
        
        # Add system message with RAG context if available
        if rag_context:
            system_msg = f"You are a helpful assistant. Use the following context to answer the user's question:\n\n{rag_context}"
            messages.append({"role": "system", "content": system_msg})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful, knowledgeable assistant."
            })
        
        # Add chat history (last 10 messages for context)
        messages.extend(chat_sessions[request.session_id][-10:])
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        # Call Groq API
        response = groq_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        assistant_message = response.choices[0].message.content
        
        # Store in chat history
        chat_sessions[request.session_id].append({
            "role": "user",
            "content": request.message
        })
        chat_sessions[request.session_id].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return ChatResponse(
            response=assistant_message,
            model_used=request.model,
            session_id=request.session_id,
            rag_used=request.use_rag and len(context_list) > 0,
            context=context_list if context_list else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
        return {"message": f"Chat history cleared for session: {session_id}"}
    return {"message": "Session not found"}


# ===================== RAG ENDPOINTS =====================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, image, or text)
    
    Supported formats: PDF, PNG, JPG, JPEG, TXT
    """
    try:
        filename = file.filename
        file_extension = filename.split('.')[-1].lower()
        
        logger.info(f"Received file: {filename} ({file_extension})")
        
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        extracted_text = None
        
        if file_extension == 'pdf':
            extracted_text = pdf_handler.extract_text_from_bytes(content)
            
        elif file_extension in ['png', 'jpg', 'jpeg']:
            extracted_text = image_handler.extract_text_from_bytes(content)
            
        elif file_extension == 'txt':
            try:
                extracted_text = content.decode('utf-8')
            except Exception as e:
                logger.error(f"Error decoding text file: {str(e)}")
                extracted_text = None
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}. Supported: PDF, PNG, JPG, JPEG, TXT"
            )
        
        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Failed to extract text from document"
            )
        
        # Add to RAG pipeline
        metadata = {
            'filename': filename,
            'file_type': file_extension
        }
        
        success = rag_pipeline.add_document(extracted_text, metadata)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add document to RAG system"
            )
        
        return UploadResponse(
            message="Document processed successfully",
            filename=filename,
            text_length=len(extracted_text),
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query uploaded documents using RAG
    
    Args:
        query: The question to ask about the documents
        k: Number of relevant chunks to retrieve (default: 3)
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        stats = rag_pipeline.vector_store.get_stats()
        if stats['total_documents'] == 0:
            raise HTTPException(
                status_code=404,
                detail="No documents uploaded. Please upload documents first."
            )
        
        results = rag_pipeline.query(request.query, k=request.k)
        
        if not results:
            return QueryResponse(
                answer="No relevant information found in the uploaded documents.",
                context=[],
                sources=[]
            )
        
        contexts = [text for text, _ in results]
        answer = f"Based on the uploaded documents:\n\n{contexts[0][:500]}..."
        
        return QueryResponse(
            answer=answer,
            context=contexts,
            sources=[f"Document chunk {i+1}" for i in range(len(contexts))]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_documents():
    """Clear all uploaded documents from RAG system"""
    try:
        rag_pipeline.clear()
        return {
            "message": "All documents cleared successfully",
            "stats": rag_pipeline.vector_store.get_stats()
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================== RESEARCH ENDPOINT =====================

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Multi-agent research workflow using CrewAI
    
    Args:
        query: Research question
        model: LLM model to use for agents
    """
    try:
        if not research_crew:
            raise HTTPException(
                status_code=503,
                detail="Research crew not initialized. Please check your environment variables (GROQ_API_KEY, SERPER_API_KEY, etc.)"
            )
        
        logger.info(f"Starting research workflow for: {request.query}")
        
        # Run the research workflow
        result = research_crew.run_simple_workflow(
            user_query=request.query,
            research_topics=[request.query]  # Can be expanded to multiple topics
        )
        
        # Extract the final result
        final_result = str(result)
        
        logger.info("Research workflow completed successfully")
        
        return ResearchResponse(
            result=final_result,
            status="completed",
            message="Research completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in research endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================== TRANSCRIPTION ENDPOINT =====================

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper
    
    Supported formats: MP3, WAV, M4A, WEBM, OGG
    """
    try:
        if not groq_client:
            raise HTTPException(
                status_code=503,
                detail="Groq API not available. Please check GROQ_API_KEY environment variable."
            )
        
        filename = audio_file.filename
        file_extension = filename.split('.')[-1].lower()
        
        # Validate audio format
        supported_formats = ['mp3', 'wav', 'm4a', 'webm', 'ogg']
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported: {', '.join(supported_formats)}"
            )
        
        logger.info(f"Received audio file: {filename}")
        
        # Read audio content
        content = await audio_file.read()
        
        # Create temporary file for Groq API
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Call Groq Whisper API
            with open(temp_file_path, "rb") as audio:
                transcription = groq_client.audio.transcriptions.create(
                    file=(filename, audio.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json"
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            return TranscriptionResponse(
                text=transcription.text,
                language=transcription.language if hasattr(transcription, 'language') else "unknown",
                duration=transcription.duration if hasattr(transcription, 'duration') else None
            )
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================== UTILITY ENDPOINTS =====================

@app.get("/models")
async def list_models():
    """List available LLM models"""
    return {
        "available_models": AVAILABLE_MODELS,
        "default_chat": "llama-3.1-8b-instant",
        "default_research": "llama-3.3-70b-versatile"
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = rag_pipeline.vector_store.get_stats()
        return {
            "rag": {
                "total_documents": stats['total_documents'],
                "embedding_dimension": stats['dimension'],
            },
            "chat": {
                "active_sessions": len(chat_sessions),
                "total_messages": sum(len(msgs) for msgs in chat_sessions.values())
            },
            "models": {
                "embedder_available": embedder_model.is_available(),
                "ocr_available": ocr_model.is_available(),
                "groq_available": groq_client is not None,
                "research_crew_available": research_crew is not None
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)