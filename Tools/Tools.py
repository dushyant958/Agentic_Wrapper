import os
from dotenv import load_dotenv
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from embedchain import App 

load_dotenv()

class Tools:
    def __init__(self, spider_api_key: str = None, firecrawl_api_key: str = None):
        # Load environment variables or passed keys
        self.spider_key = spider_api_key or os.getenv("SPIDER_API_KEY")
        self.firecrawl_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.hfkey = os.getenv("HF_TOKEN")
        self.serper_key = os.getenv("SERPER_API_KEY")

        # ✅ Create the EmbedChain App directly (new API)
        try:
            self.embedchain = App(
                config={
                    "embedding_model": "hkunlp/instructor-xl",  # HuggingFace model
                    "hf_api_key": self.hfkey,
                    "persist_directory": "./vector_store"
                }
            )
        except Exception as e:
            print(f"Warning: EmbedChain initialization failed: {e}")
            self.embedchain = None

        # Initialize scraping tools - REMOVED embedder parameter
        self.tool1 = self.scrape_tool()
        self.tool2 = self.serper_tool()
        self.tool3 = self.scrape_tool()  # Using basic scraper as tool3

    def scrape_tool(self) -> ScrapeWebsiteTool:
        """
        Returns a ScrapeWebsiteTool instance that can scrape any website dynamically.
        Note: Removed embedder parameter as it's not supported by CrewAI tools
        """
        return ScrapeWebsiteTool()

    def serper_tool(self) -> SerperDevTool:
        """
        Returns a SerperDevTool for web search capabilities.
        Requires SERPER_API_KEY in environment variables.
        """
        if self.serper_key:
            return SerperDevTool()
        else:
            print("Warning: SERPER_API_KEY not found. Search functionality may be limited.")
            return SerperDevTool()

    def store_in_embedchain(self, url: str):
        """
        Manually add a URL to the embedchain vector store.
        Use this after scraping to store the data.
        """
        if self.embedchain:
            try:
                self.embedchain.add(url)
                print(f"✓ Successfully added {url} to vector store")
            except Exception as e:
                print(f"✗ Failed to add {url} to vector store: {e}")
        else:
            print("EmbedChain not initialized")

    def query_embeddings(self, query: str, top_k: int = 5):
        """
        Retrieve top-k most relevant documents from the vector store for a given query.
        """
        if self.embedchain:
            try:
                return self.embedchain.query(query)
            except Exception as e:
                print(f"Error querying embeddings: {e}")
                return None
        else:
            print("EmbedChain not initialized")
            return None

    def add_text_to_store(self, text: str):
        """
        Add raw text to the vector store.
        Useful for storing scraped content.
        """
        if self.embedchain:
            try:
                self.embedchain.add_local("text", text)
                print(f"✓ Successfully added text to vector store")
            except Exception as e:
                print(f"✗ Failed to add text to vector store: {e}")
        else:
            print("EmbedChain not initialized")