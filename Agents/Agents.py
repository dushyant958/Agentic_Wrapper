import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from Tools.Tools import Tools

load_dotenv()

os.environ["CREWAI_LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ADDING THIS LINE TO PREVENT OPENAI EMBEDDING REQUIREMENT:
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-not-used"  # Dummy key to prevent errors


class Agents:
    def __init__(self, groq_api_key: str = None, serper_api_key: str = None, llm_choice: str = None, embedder_config: dict = None):
        self.embedder_config = embedder_config or {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.serper_key = serper_api_key or os.getenv("SERPER_API_KEY")

        # Validate GROQ API key exists
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY is required. Please set it in your .env file.")

        self.llm_instance = self.initialise_llms(llm_choice)
        self.tools = Tools()
        self.agent1 = self.planner_agent()
        self.agent2 = self.research_agent()
        self.agent3 = self.verifier_agent()
        self.agent4 = self.writer_agent()

    def initialise_llms(self, llm_choice: str = None):
        llm_choice = llm_choice or "llama-3.3-70b-versatile"
        llm_options = {
            "llama-3.1-8b-instant": lambda: ChatGroq(
                groq_api_key=self.groq_key, 
                model="llama-3.1-8b-instant",
                temperature=0.7
            ),
            "llama-3.3-70b-versatile": lambda: ChatGroq(
                groq_api_key=self.groq_key, 
                model="llama-3.3-70b-versatile",
                temperature=0.7
            ),
            "meta-llama/llama-guard-4-12b": lambda: ChatGroq(
                groq_api_key=self.groq_key, 
                model="meta-llama/llama-guard-4-12b",
                temperature=0.7
            )
        }  
        if llm_choice not in llm_options:
            raise ValueError(f"Invalid choice of LLM: {llm_choice}. Available options are: {list(llm_options.keys())}")

        return llm_options[llm_choice]()  
    
    def planner_agent(self):
        groq_api_key = self.groq_key
        agent = Agent(
            role="Task Planner Agent",
            goal="Analyze the user's query or task and break it down into smaller, manageable sub-tasks. "
                 "For each sub-task, clearly define: (1) the objective, (2) the specific steps required, "
                 "and (3) the expected outcome. Present the results as a structured execution plan.",
            backstory="With years of experience as a strategic consultant and project manager, you have mastered "
                      "the art of turning complex problems into clear, actionable roadmaps. You are relied upon "
                      "for your ability to analyze ambiguous tasks, break them into structured sub-goals, and "
                      "provide execution-ready plans. Your analytical thinking, attention to detail, and "
                      "systematic approach ensure that every plan is both realistic and results-driven.",
            llm=self.llm_instance,
            verbose=True,
            allow_delegation=True,
            max_iter=5,
            max_execution_time=200,
            max_retry_limit=3,
            multimodal=True,
            respect_context_window=True,
            reasoning=True,
            max_reasoning_attempts=2,
            use_system_prompt=True,
            max_rpm=None,
            knowledge_sources=None,
            embedder=self.embedder_config,  # ADD THIS LINE
            memory=True
        )
        return agent
 
    def research_agent(self):
        agent = Agent(
            role="Research Agent",
            goal="Gather high-quality, relevant information to support the planner's sub-tasks. "
                "Summarize findings clearly, cite sources, and filter out irrelevant content.",
            backstory="As an expert researcher with years of experience in information retrieval and data analysis, "
                    "you excel at quickly finding accurate and up-to-date information. You know how to identify "
                    "trustworthy sources, summarize key points, and organize data in a way that is immediately useful "
                    "for task planning or decision-making.",
            llm=self.llm_instance,
            verbose=True,
            allow_delegation=True,
            max_iter=5,
            max_execution_time=200,
            max_retry_limit=3,
            multimodal=True,
            respect_context_window=True,
            reasoning=True,
            max_reasoning_attempts=2,
            use_system_prompt=True,
            max_rpm=None,
            knowledge_sources=None,
            embedder=self.embedder_config,
            memory=True,
            tools=[self.tools.tool1, self.tools.tool2, self.tools.tool3]
        )
        return agent

    def verifier_agent(self):
        agent = Agent(
            role="Verification Agent",
            goal="Check the accuracy, consistency, and reliability of the information gathered by research agents. "
                 "Flag inconsistencies and provide validation summaries for each sub-task.",
            backstory="As a meticulous fact-checker and quality assurance specialist, you have extensive experience in "
                      "verifying complex data and ensuring its correctness. You identify discrepancies, validate sources, "
                      "and ensure that all conclusions drawn are supported by evidence.",
            llm=self.llm_instance,
            verbose=True,
            allow_delegation=True,
            max_iter=5,
            max_execution_time=200,
            max_retry_limit=3,
            multimodal=True,
            respect_context_window=True,
            reasoning=True,
            max_reasoning_attempts=2,
            use_system_prompt=True,
            max_rpm=None,
            knowledge_sources=None,
            embedder=self.embedder_config,
            memory=True,
            tools=[self.tools.tool1, self.tools.tool2, self.tools.tool3]
        )
        return agent

    def writer_agent(self):
        agent = Agent(
            role="Writer Agent",
            goal="Take verified information and structured plans and synthesize them into clear, concise, "
                 "and actionable outputs such as reports, instructions, or documentation.",
            backstory="As a skilled technical and creative writer, you specialize in converting complex information "
                      "into readable, actionable, and well-structured content. You ensure clarity, coherence, and "
                      "effectiveness in all written outputs, making them easy for humans or AI systems to follow.",
            llm=self.llm_instance,
            verbose=True,
            allow_delegation=True,
            max_iter=5,
            max_execution_time=200,
            max_retry_limit=3,
            multimodal=True,
            respect_context_window=True,
            reasoning=True,
            max_reasoning_attempts=2,
            use_system_prompt=True,
            max_rpm=None,
            knowledge_sources=None,
            embedder=self.embedder_config,
            memory=True
        )
        return agent