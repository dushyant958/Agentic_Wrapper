"""
Image Handler for OCR text extraction from images
"""
import easyocr
from typing import Optional, List
import logging
import numpy as np
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles image OCR with robust error handling"""
    
    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize OCR reader
        
        Args:
            languages: List of language codes for OCR (default: English)
        """
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
            logger.info(f"EasyOCR initialized with languages: {languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            self.reader = None
    
    def extract_text_from_path(self, image_path: str) -> Optional[str]:
        """
        Extract text from image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        if not self.reader:
            logger.error("OCR reader not initialized")
            return None
        
        try:
            result = self.reader.readtext(image_path, detail=0)
            
            if not result:
                logger.warning(f"No text detected in image {image_path}")
                return None
            
            text = " ".join(result)
            logger.info(f"Successfully extracted {len(text)} characters from image")
            return text
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[str]:
        """
        Extract text from image bytes (for uploaded files)
        
        Args:
            image_bytes: Image file content as bytes
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        if not self.reader:
            logger.error("OCR reader not initialized")
            return None
        
        try:
            # Convert bytes to numpy array for EasyOCR
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            result = self.reader.readtext(image_np, detail=0)
            
            if not result:
                logger.warning("No text detected in image")
                return None
            
            text = " ".join(result)
            logger.info(f"Successfully extracted {len(text)} characters from image")
            return text
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """Check if OCR reader is properly initialized"""
        return self.reader is not None