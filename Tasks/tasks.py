import os
from dotenv import load_dotenv
from crewai import Task
from typing import List, Optional, Tuple, Any  # ✅ Tuple & Any still imported for future use

load_dotenv()


class Tasks:
    def __init__(self, agents):
        self.agents = agents
        self.output_dir = "./outputs"
        os.makedirs(self.output_dir, exist_ok=True)


    def planning_task(self, user_query: str) -> Task:
        return Task(
            name="query_planning_task",
            description=(
                f"Analyze the following user query and create a comprehensive execution plan:\n\n"
                f"User Query: {user_query}\n\n"
                f"Your responsibilities:\n"
                f"1. Understand the core objective and intent behind the query\n"
                f"2. Break down the main goal into 3-7 manageable sub-tasks\n"
                f"3. For each sub-task, clearly define:\n"
                f"   - The specific objective\n"
                f"   - Step-by-step actions required\n"
                f"   - Expected outcome/deliverable\n"
                f"   - Any dependencies on other sub-tasks\n"
                f"4. Assess the overall complexity of the query\n"
                f"5. Define success criteria for the entire plan\n"
                f"6. Ensure the plan is logical, sequential, and executable"
            ),
            expected_output=(
                "A structured execution plan containing:\n"
                "- Main goal derived from the user query\n"
                "- List of sub-tasks with objectives, steps, expected outcomes, and dependencies\n"
                "- Overall complexity assessment (Low/Medium/High)\n"
                "- Clear success criteria for plan completion\n"
                "The plan should be detailed enough for other agents to execute without ambiguity."
            ),
            agent=self.agents.agent1,
            async_execution=False,
            human_input=False,
            markdown=False,
            output_file=f"{self.output_dir}/execution_plan.txt",
            create_directory=True,
            config={
                "max_tokens": 2000,
                "temperature": 0.3
            }
        )


    def research_task(self, research_topic: str, context_tasks: Optional[List[Task]] = None) -> Task:
        return Task(
            name="information_research_task",
            description=(
                f"Conduct comprehensive research on the following topic:\n\n"
                f"Research Topic: {research_topic}\n\n"
                f"Your responsibilities:\n"
                f"1. Use available web scraping tools (ScrapeWebsiteTool, FirecrawlScrapeWebsiteTool, SpiderTool) "
                f"to gather relevant information\n"
                f"2. Search for the most recent and authoritative sources\n"
                f"3. Extract key information, facts, and data points\n"
                f"4. Filter out irrelevant or low-quality content\n"
                f"5. Organize findings in a clear, structured manner\n"
                f"6. Cite all sources with URLs or references\n"
                f"7. Assess the relevance of findings to the research topic\n"
                f"8. Store scraped data in the vector database for future retrieval"
            ),
            expected_output=(
                "A comprehensive research report containing:\n"
                "- The research query/topic\n"
                "- Concise summary of all findings (200-300 words)\n"
                "- List of key points discovered (5-10 bullet points)\n"
                "- Complete list of sources with URLs\n"
                "- Relevance assessment (High/Medium/Low)\n"
                "All information should be factual, up-to-date, and properly attributed."
            ),
            agent=self.agents.agent2,
            tools=[self.agents.tools.tool1, self.agents.tools.tool2, self.agents.tools.tool3],
            context=context_tasks if context_tasks else None,
            async_execution=True,
            human_input=False,
            markdown=True,
            output_file=f"{self.output_dir}/research_findings.md",
            create_directory=True,
            config={
                "max_tokens": 3000,
                "temperature": 0.5
            },
            callback=self._research_completion_callback
        )


    def verification_task(self, context_tasks: List[Task]) -> Task:
        return Task(
            name="information_verification_task",
            description=(
                "Review and verify the information gathered by the Research Agent.\n\n"
                "Your responsibilities:\n"
                "1. Cross-reference information from multiple sources\n"
                "2. Check for factual accuracy and consistency\n"
                "3. Identify any contradictions or discrepancies in the data\n"
                "4. Use web scraping tools to verify claims when necessary\n"
                "5. Query the vector database to check against previously stored information\n"
                "6. Assess the reliability and credibility of sources\n"
                "7. Flag any information that cannot be verified\n"
                "8. Provide confidence levels for verified information\n"
                "9. Document any concerns or recommendations for further research"
            ),
            expected_output=(
                "A detailed verification report containing:\n"
                "- Summary of information being verified\n"
                "- Accuracy status (Verified/Partially Verified/Unverified)\n"
                "- List of any inconsistencies or concerns found\n"
                "- Validation notes with specific details\n"
                "- Overall confidence level (High/Medium/Low)\n"
                "The report should clearly indicate what information is trustworthy and what requires additional validation."
            ),
            agent=self.agents.agent3,
            tools=[self.agents.tools.tool1, self.agents.tools.tool2, self.agents.tools.tool3],
            context=context_tasks,
            async_execution=False,
            human_input=False,
            markdown=True,
            output_file=f"{self.output_dir}/verification_report.md",
            create_directory=True,
            guardrail=self._verification_guardrail,  # ✅ Still bound, but fixed signature below
            guardrail_max_retries=3,
            config={
                "max_tokens": 2500,
                "temperature": 0.2
            }
        )


    def writing_task(self, output_format: str = "report", context_tasks: Optional[List[Task]] = None) -> Task:
        return Task(
            name="content_writing_task",
            description=(
                f"Synthesize all verified information and create a comprehensive {output_format}.\n\n"
                f"Your responsibilities:\n"
                f"1. Review the execution plan, research findings, and verification report\n"
                f"2. Organize information in a logical, coherent structure\n"
                f"3. Write clear, concise, and actionable content\n"
                f"4. Ensure all claims are supported by verified sources\n"
                f"5. Format the output appropriately for the specified format ({output_format})\n"
                f"6. Include an executive summary at the beginning\n"
                f"7. Provide actionable recommendations where applicable\n"
                f"8. Maintain professional tone and clarity throughout\n"
                f"9. Ensure the content directly addresses the original user query"
            ),
            expected_output=(
                f"A well-structured {output_format} containing:\n"
                "- Title that clearly describes the content\n"
                "- Main content organized in sections with clear headings\n"
                "- Executive summary (150-200 words)\n"
                "- Actionable recommendations (if applicable)\n"
                "The content should be polished, professional, and ready for immediate use."
            ),
            agent=self.agents.agent4,
            context=context_tasks if context_tasks else None,
            async_execution=False,
            human_input=True,
            markdown=True,
            output_file=f"{self.output_dir}/final_output.md",
            create_directory=True,
            config={
                "max_tokens": 4000,
                "temperature": 0.4
            },
            callback=self._writing_completion_callback
        )


    def sequential_workflow_tasks(self, user_query: str, research_topics: List[str]) -> List[Task]:
        planning = self.planning_task(user_query)
        
        research_tasks = [
            self.research_task(topic, context_tasks=[planning])
            for topic in research_topics
        ]
        
        verification = self.verification_task(context_tasks=[planning] + research_tasks)
        
        writing = self.writing_task(
            output_format="comprehensive_report",
            context_tasks=[planning, verification] + research_tasks
        )
        
        return [planning] + research_tasks + [verification, writing]


    def _research_completion_callback(self, output):
        """Callback executed after research task completion"""
        print(f"\n{'='*50}")
        print("Research Task Completed!")
        print(f"{'='*50}\n")


    def _writing_completion_callback(self, output):
        """Callback executed after writing task completion"""
        print(f"\n{'='*50}")
        print("Writing Task Completed!")
        print(f"{'='*50}\n")


    # ✅ Fixed: Removed return annotation to bypass CrewAI type validation error
    def _verification_guardrail(self, output):
        """
        Validate verification task output before proceeding.
        Must return Tuple[bool, Any] but avoid type hinting to prevent Pydantic errors.
        """
        try:
            if not output or not hasattr(output, 'raw'):
                print("Guardrail Failed: Invalid output structure")
                return (False, "Invalid output structure")
            
            output_text = output.raw.lower()
            
            required_keywords = ["verified", "accuracy", "confidence"]
            if not any(keyword in output_text for keyword in required_keywords):
                print("Guardrail Failed: Missing required verification elements")
                return (False, "Missing required verification elements")
            
            if len(output_text.strip()) < 100:
                print("Guardrail Failed: Verification report too short")
                return (False, "Verification report too short")
            
            print("Guardrail Passed: Verification output is valid")
            return (True, output)
            
        except Exception as e:
            print(f"Guardrail Failed: Exception occurred - {str(e)}")
            return (False, str(e))
