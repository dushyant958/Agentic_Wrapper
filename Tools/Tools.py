import os
from dotenv import load_dotenv
from crewai_tools import ScrapeWebsiteTool, SpiderTool, FirecrawlScrapeWebsiteTool
from embedchain.config import AppConfig
from embedchain.embedchain import EmbedChain
load_dotenv()

class Tools:
    def __init__(self, spider_api_key: str = None, firecrawl_api_key: str = None):
        self.spider_key = spider_api_key or os.getenv("SPIDER_API_KEY")
        self.firecrawl_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        self.hfkey = os.getenv("HF_TOKEN")

       
        self.embed_config = AppConfig(
            embedding_model="hkunlp/instructor-xl",  # HuggingFace model
            hf_api_key=self.hfkey,
            persist_directory="./vector_store"       
        )
        self.embedchain = EmbedChain(config=self.embed_config)

        self.tool1 = self.scrape_tool()
        self.tool2 = self.firecrawl_tool()
        self.tool3 = self.spider_tool()

    def scrape_tool(self) -> ScrapeWebsiteTool:
        """
        Returns a ScrapeWebsiteTool instance that can scrape any website dynamically.
        Data retrieved is automatically embedded into the local vector store.
        """
        return ScrapeWebsiteTool(embedder=self.embedchain)

    def firecrawl_tool(self) -> FirecrawlScrapeWebsiteTool:
        """
        Returns a FirecrawlScrapeWebsiteTool instance that can scrape any website dynamically.
        Requires FIRECRAWL_API_KEY for authentication.
        """
        return FirecrawlScrapeWebsiteTool(api_key=self.firecrawl_key, embedder=self.embedchain)

    def spider_tool(self) -> SpiderTool:
        """
        Returns a SpiderTool instance that can scrape any website dynamically.
        Requires SPIDER_API_KEY for authentication.
        """
        return SpiderTool(api_key=self.spider_key, embedder=self.embedchain)

    def query_embeddings(self, query: str, top_k: int = 5):
        """
        Retrieve top-k most relevant documents from the vector store for a given query.
        """
        return self.embedchain.query(query, top_k=top_k)
