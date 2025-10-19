"""
Whisper Model for audio transcription using Groq API
"""
from groq import Groq
from typing import Optional, Dict
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperModel:
    """Whisper transcription using Groq API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Whisper model
        
        Args:
            api_key: Groq API key (reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Whisper model initialized with Groq API")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                self.client = None
    
    def transcribe_file(
        self, 
        file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Transcribe audio file
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'es')
            prompt: Optional prompt to guide transcription
            
        Returns:
            Dict with transcription results or None if failed
        """
        if not self.client:
            logger.error("Groq client not initialized")
            return None
        
        try:
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), audio_file.read()),
                    model="whisper-large-v3",
                    language=language,
                    prompt=prompt,
                    response_format="verbose_json",
                    temperature=0.0
                )
            
            result = {
                "text": transcription.text,
                "language": transcription.language if hasattr(transcription, 'language') else "unknown",
                "duration": transcription.duration if hasattr(transcription, 'duration') else None,
                "segments": transcription.segments if hasattr(transcription, 'segments') else []
            }
            
            logger.info(f"Transcription successful: {len(result['text'])} characters")
            return result
            
        except FileNotFoundError:
            logger.error(f"Audio file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Transcribe audio from bytes
        
        Args:
            audio_bytes: Audio file content as bytes
            filename: Original filename (for format detection)
            language: Language code
            prompt: Optional prompt
            
        Returns:
            Dict with transcription results or None if failed
        """
        if not self.client:
            logger.error("Groq client not initialized")
            return None
        
        try:
            # Create temporary file
            file_extension = filename.split('.')[-1] if '.' in filename else 'mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Transcribe
            result = self.transcribe_file(temp_file_path, language, prompt)
            
            # Clean up
            os.unlink(temp_file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """Check if Whisper API is available"""
        return self.client is not None