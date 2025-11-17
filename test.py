def test_imports():
    print("Testing imports...")
    
    try:
        from langchain_core.utils.function_calling import convert_to_openai_function
        print("✓ LangChain core imports OK")
    except ImportError as e:
        print(f"✗ LangChain core error: {e}")
    
    try:
        from langchain_community.docstore.document import Document
        print("✓ LangChain community imports OK")
    except ImportError as e:
        print(f"✗ LangChain community error: {e}")
    
    try:
        from crewai import Agent, Task, Crew
        print("✓ CrewAI imports OK")
    except ImportError as e:
        print(f"✗ CrewAI error: {e}")
    
    try:
        from embedchain import App
        print("✓ Embedchain imports OK")
    except ImportError as e:
        print(f"✗ Embedchain error: {e}")
    
    try:
        import faiss
        print("✓ FAISS imports OK")
    except ImportError as e:
        print(f"✗ FAISS error: {e}")
    
    try:
        import easyocr
        print("✓ EasyOCR imports OK")
    except ImportError as e:
        print(f"✗ EasyOCR error: {e}")
    
    try:
        import pdfplumber
        print("✓ PDFPlumber imports OK")
    except ImportError as e:
        print(f"✗ PDFPlumber error: {e}")
    
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imports OK")
    except ImportError as e:
        print(f"✗ FastAPI error: {e}")
    
    try:
        import streamlit
        print("✓ Streamlit imports OK")
    except ImportError as e:
        print(f"✗ Streamlit error: {e}")
    
    try:
        from groq import Groq
        print("✓ Groq imports OK")
    except ImportError as e:
        print(f"✗ Groq error: {e}")
    
    print("\nAll critical imports tested!")

if __name__ == "__main__":
    test_imports()