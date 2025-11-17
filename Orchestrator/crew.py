import os
from dotenv import load_dotenv
from crewai import Crew, Process
from typing import List, Dict, Any, Optional
from Agents.Agents import Agents
from Tasks.Tasks import Tasks

load_dotenv()


class ResearchCrew:
    def __init__(self, llm_choice: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Research Crew with agents and tasks.
        
        Args:
            llm_choice: The LLM model to use for all agents
        """
        self.agents = Agents(llm_choice=llm_choice)
        self.tasks = Tasks(self.agents)
        self.crew = None


    def create_crew(
        self, 
        process_type: Process = Process.sequential,
        verbose: bool = True,
        memory: bool = True,
        cache: bool = True,
        max_rpm: int = 100,
        share_crew: bool = False
    ) -> Crew:
        """
        Create and configure the Crew with all agents and default settings.
        
        Args:
            process_type: The process type (sequential or hierarchical)
            verbose: Enable verbose output
            memory: Enable memory across executions
            cache: Enable caching of results
            max_rpm: Maximum requests per minute
            share_crew: Share crew performance data with CrewAI
        
        Returns:
            Configured Crew instance
        """
        self.crew = Crew(
            agents=[
                self.agents.agent1,  # Planner
                self.agents.agent2,  # Researcher
                self.agents.agent3,  # Verifier
                self.agents.agent4   # Writer
            ],
            tasks=[],  # Tasks will be added dynamically
            process=process_type,
            verbose=verbose,
            memory=memory,
            cache=cache,
            max_rpm=max_rpm,
            share_crew=share_crew,
            full_output=True,
            step_callback=self._step_callback
        )
        
        return self.crew


    def run_simple_workflow(self, user_query: str, research_topics: List[str]) -> Dict[str, Any]:
        """
        Runs a simple sequential workflow with planning, research, verification, and writing.
        
        Args:
            user_query: The main query from the user
            research_topics: List of topics to research
        
        Returns:
            Dictionary containing the final result and task outputs
        """
        print(f"\n{'='*60}")
        print(f"Starting Research Crew Workflow")
        print(f"Query: {user_query}")
        print(f"Research Topics: {', '.join(research_topics)}")
        print(f"{'='*60}\n")
        
        # Create tasks for the workflow
        workflow_tasks = self.tasks.sequential_workflow_tasks(user_query, research_topics)
        
        # Create crew with these tasks
        crew = Crew(
            agents=[
                self.agents.agent1,
                self.agents.agent2,
                self.agents.agent3,
                self.agents.agent4
            ],
            tasks=workflow_tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            cache=True,
            max_rpm=100,
            share_crew=False,
            full_output=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        print(f"\n{'='*60}")
        print("Workflow Completed Successfully!")
        print(f"{'='*60}\n")
        
        return result


    # def run_custom_workflow(
    #     self, 
    #     user_query: str, 
    #     custom_tasks: List[Any],
    #     process_type: Process = Process.sequential
    # ) -> Dict[str, Any]:
    #     """
    #     Run a custom workflow with user-defined tasks.
        
    #     Args:
    #         user_query: The main query from the user
    #         custom_tasks: List of custom Task objects
    #         process_type: Sequential or hierarchical process
        
    #     Returns:
    #         Dictionary containing the final result and task outputs
    #     """
    #     print(f"\n{'='*60}")
    #     print(f"Starting Custom Workflow")
    #     print(f"Query: {user_query}")
    #     print(f"Number of Tasks: {len(custom_tasks)}")
    #     print(f"{'='*60}\n")
        
    #     crew = Crew(
    #         agents=[
    #             self.agents.agent1,
    #             self.agents.agent2,
    #             self.agents.agent3,
    #             self.agents.agent4
    #         ],
    #         tasks=custom_tasks,
    #         process=process_type,
    #         verbose=True,
    #         memory=True,
    #         cache=True,
    #         max_rpm=100,
    #         share_crew=False,
    #         full_output=True
    #     )
        
    #     result = crew.kickoff()
        
    #     print(f"\n{'='*60}")
    #     print("Custom Workflow Completed!")
    #     print(f"{'='*60}\n")
        
    #     return result


    def run_planning_only(self, user_query: str) -> Dict[str, Any]:
        """
        Run only the planning task to break down a query.
        Useful for getting a plan before executing research.
        
        Args:
            user_query: The query to analyze and plan
        
        Returns:
            Planning result
        """
        print(f"\n{'='*60}")
        print(f"Running Planning Task Only")
        print(f"Query: {user_query}")
        print(f"{'='*60}\n")
        
        planning_task = self.tasks.planning_task(user_query)
        
        crew = Crew(
            agents=[self.agents.agent1],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=True,
            max_rpm=100,
            share_crew=False
        )
        
        result = crew.kickoff()
        
        print(f"\n{'='*60}")
        print("Planning Completed!")
        print(f"{'='*60}\n")
        
        return result


    def run_research_only(self, research_topic: str) -> Dict[str, Any]:
        """
        Run only research task for a specific topic.
        Useful for quick information gathering.
        
        Args:
            research_topic: The topic to research
        
        Returns:
            Research result
        """
        print(f"\n{'='*60}")
        print(f"Running Research Task Only")
        print(f"Topic: {research_topic}")
        print(f"{'='*60}\n")
        
        research_task = self.tasks.research_task(research_topic)
        
        crew = Crew(
            agents=[self.agents.agent2],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            cache=True,
            max_rpm=100,
            share_crew=False
        )
        
        result = crew.kickoff()
        
        print(f"\n{'='*60}")
        print("Research Completed!")
        print(f"{'='*60}\n")
        
        return result


    def _step_callback(self, step_output):
        """
        Callback function executed after each step in the crew process.
        Useful for monitoring and logging.
        """
        print(f"\n--- Step Completed ---")
        print(f"Output: {str(step_output)[:200]}...")  # Print first 200 chars
        print(f"--- End of Step ---\n")


    def get_crew_info(self) -> Dict[str, Any]:
        """
        Get information about the current crew configuration.
        
        Returns:
            Dictionary with crew information
        """
        if not self.crew:
            return {"status": "Crew not initialized"}
        
        return {
            "total_agents": len(self.crew.agents),
            "total_tasks": len(self.crew.tasks),
            "process_type": self.crew.process,
            "memory_enabled": self.crew.memory,
            "cache_enabled": self.crew.cache,
            "agents": [
                {
                    "role": agent.role,
                    "goal": agent.goal[:100] + "..." if len(agent.goal) > 100 else agent.goal
                }
                for agent in self.crew.agents
            ]
        }


# Main execution example
if __name__ == "__main__":
    # Initialize the crew
    research_crew = ResearchCrew(llm_choice="llama-3.3-70b-versatile")
    
    # Example 1: Run a complete workflow
    user_query = "What are the latest developments in AI safety and alignment?"
    research_topics = [
        "AI safety research 2024",
        "AI alignment techniques",
        "Current AI safety challenges"
    ]
    
    result = research_crew.run_simple_workflow(
        user_query=user_query,
        research_topics=research_topics
    )
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(result)
    
    # Example 2: Run planning only
    # planning_result = research_crew.run_planning_only(
    #     user_query="How can I build a machine learning pipeline?"
    # )
    # print(planning_result)
    
    # Example 3: Run research only
    # research_result = research_crew.run_research_only(
    #     research_topic="Latest trends in generative AI"
    # )
    # print(research_result)