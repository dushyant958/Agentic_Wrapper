"""
OCR Model wrapper - simple approach with auto-caching
"""
import easyocr
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRModel:
    """Simple OCR model wrapper with automatic caching"""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize OCR Model
        
        Args:
            languages: List of language codes (default: English)
            gpu: Whether to use GPU acceleration
        """
        try:
            logger.info(f"Initializing EasyOCR with languages: {languages}")
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            self.reader = None
    
    def is_available(self) -> bool:
        """Check if OCR is available"""
        return self.reader is not None