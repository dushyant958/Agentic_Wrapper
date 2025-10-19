"""
PDF Handler for extracting text from PDF files
"""
import pdfplumber
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFHandler:
    """Handles PDF text extraction with robust error handling"""
    
    @staticmethod
    def extract_text(file_path: str) -> Optional[str]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            text_content = []
            
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) == 0:
                    logger.warning(f"PDF {file_path} has no pages")
                    return None
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                        else:
                            logger.warning(f"No text found on page {page_num}")
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {str(e)}")
                        continue
            
            if not text_content:
                logger.warning(f"No text extracted from PDF {file_path}")
                return None
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except FileNotFoundError:
            logger.error(f"PDF file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes) -> Optional[str]:
        """
        Extract text from PDF bytes (for uploaded files)
        
        Args:
            file_bytes: PDF file content as bytes
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            import io
            text_content = []
            
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if len(pdf.pages) == 0:
                    logger.warning("PDF has no pages")
                    return None
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {str(e)}")
                        continue
            
            if not text_content:
                logger.warning("No text extracted from PDF")
                return None
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"Error processing PDF bytes: {str(e)}")
            return None