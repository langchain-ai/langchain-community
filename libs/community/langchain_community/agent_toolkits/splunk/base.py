"""Splunk agent creation and utilities."""

from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from langchain_community.agent_toolkits.splunk.toolkit import SplunkToolkit
from langchain_community.utilities.splunk import SplunkAPIWrapper


# Default prompt templates for Splunk SPL agent
SPLUNK_AGENT_PREFIX = """You are an expert Splunk SPL (Search Processing Language) analyst.
You have access to tools that allow you to:
1. Get information about available Splunk indexes, sourcetypes, and hosts
2. Execute SPL queries against Splunk data
3. Validate SPL query syntax and performance
4. List available data sources

When working with Splunk data, follow these best practices:
- Always start by getting information about available data using the info tools
- Write efficient SPL queries with appropriate time ranges when possible
- Use proper SPL syntax: start with 'search' for basic searches or use pipe commands
- Specify indexes when known to improve performance
- Be mindful of query performance - avoid expensive operations like joins when possible
- Provide clear explanations of your queries and results

IMPORTANT SPL SYNTAX REMINDERS:
- Basic search: search index=main error
- With time range: search index=main error earliest=-1h@h latest=now  
- Statistical analysis: search index=main | stats count by sourcetype
- Field extraction: search index=main | eval new_field=if(status=200,"success","error")
- Filtering: search index=main | where status>400
- Sorting: search index=main error | sort -_time | head 10

Available tools:"""

SPLUNK_AGENT_FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SPLUNK_AGENT_SUFFIX = """Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def create_splunk_agent(
    llm: BaseLanguageModel,
    toolkit: SplunkToolkit,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create a Splunk SPL agent from LLM and toolkit.

    Args:
        llm: Language model to use for the agent
        toolkit: Splunk toolkit with configured tools
        agent_executor_kwargs: Additional arguments for AgentExecutor
        **kwargs: Additional arguments (e.g., verbose)

    Returns:
        AgentExecutor configured for Splunk SPL queries
    """
    tools = toolkit.get_tools()

    # Create prompt template
    prompt = PromptTemplate(
        template=SPLUNK_AGENT_PREFIX
        + "\n\n{tools}\n\n"
        + SPLUNK_AGENT_FORMAT_INSTRUCTIONS
        + "\n\n"
        + SPLUNK_AGENT_SUFFIX,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools]),
        },
    )

    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Set up agent executor arguments
    executor_kwargs = agent_executor_kwargs or {}

    # Create and return agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=kwargs.get("verbose", False),
        handle_parsing_errors=True,
        max_iterations=kwargs.get("max_iterations", 15),
        max_execution_time=kwargs.get("max_execution_time", 60),
        **executor_kwargs,
    )


def create_splunk_agent_from_api_wrapper(
    llm: BaseLanguageModel,
    splunk_wrapper: SplunkAPIWrapper,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create a Splunk SPL agent directly from API wrapper.

    Args:
        llm: Language model to use for the agent
        splunk_wrapper: Configured Splunk API wrapper
        agent_executor_kwargs: Additional arguments for AgentExecutor
        **kwargs: Additional arguments

    Returns:
        AgentExecutor configured for Splunk SPL queries
    """
    toolkit = SplunkToolkit(splunk_wrapper=splunk_wrapper, llm=llm)
    return create_splunk_agent(
        llm=llm, toolkit=toolkit, agent_executor_kwargs=agent_executor_kwargs, **kwargs
    )
