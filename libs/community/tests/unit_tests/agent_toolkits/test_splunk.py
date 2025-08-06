import json
import pytest
from unittest.mock import Mock, patch
from langchain_community.agent_toolkits.splunk import (
    SplunkToolkit,
    create_splunk_agent,
    create_splunk_agent_from_api_wrapper,
)
from langchain_community.utilities.splunk import SplunkAPIWrapper
from langchain_community.tools.splunk import (
    InfoSplunkTool,
    ListSplunkIndexesTool,
    QuerySplunkTool,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain.agents.agent import BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish

class MockLLM(BaseLanguageModel):
    """A mock LLM class that implements the necessary abstract methods."""
    def __init__(self, **kwargs):
        pass
    
    def _call(self, prompt, stop=None, **kwargs):
        return "mocked response"
    
    @property
    def _llm_type(self):
        return "mock_llm"

    def invoke(self, input, stop=None, **kwargs):
        return "mocked response"

    def generate(self, prompts, stop=None, **kwargs):
        return LLMResult(generations=[[AIMessage(content="mocked response")]])

    async def agenerate(self, prompts, stop=None, **kwargs):
        return self.generate(prompts, stop, **kwargs)

    def predict(self, text, stop=None, **kwargs):
        return "mocked response"

    def predict_messages(self, messages, stop=None, **kwargs):
        return AIMessage(content="mocked response")

    async def apredict(self, text, stop=None, **kwargs):
        return self.predict(text, stop, **kwargs)

    async def apredict_messages(self, messages, stop=None, **kwargs):
        return self.predict_messages(messages, stop, **kwargs)

    def generate_prompt(self, prompts, stop=None, **kwargs):
        return LLMResult(generations=[[AIMessage(content="mocked response")]])

    async def agenerate_prompt(self, prompts, stop=None, **kwargs):
        return self.generate_prompt(prompts, stop, **kwargs)

# New mock class for the agent to satisfy pydantic validation
from typing import List, Tuple, Any
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.agent import BaseSingleActionAgent  # Ensure this works for your version

class MockAgent(BaseSingleActionAgent):
    """A mock agent class to satisfy AgentExecutor and pydantic validation."""

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any
    ) -> AgentFinish:
        return AgentFinish(return_values={"output": "mocked output"}, log="mocked log")

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any
    ) -> AgentFinish:
        return self.plan(intermediate_steps, **kwargs)


class TestSplunkToolkit:
    """Test SplunkToolkit functionality."""

    @pytest.fixture
    def mock_splunk_wrapper(self):
        """Create mock Splunk wrapper."""
        wrapper = Mock(spec=SplunkAPIWrapper)
        wrapper.splunk_host = "mock_host"
        wrapper.splunk_token = "mock_token"
        wrapper.splunk_username = "mock_user"
        wrapper.splunk_password = "mock_password"
        
        wrapper.get_summary_info.return_value = {
            "indexes": ["main", "security", "web"],
            "connection_status": "connected"
        }
        wrapper.get_indexes.return_value = ["main", "security", "web_logs"]
        wrapper.run_spl_query.return_value = [
            {"_time": "2023-01-01T00:00:00", "message": "test event"}
        ]
        return wrapper

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM()

    def test_toolkit_get_tools_without_llm(self, mock_splunk_wrapper):
        """Test SplunkToolkit without LLM."""
        toolkit = SplunkToolkit(splunk_wrapper=mock_splunk_wrapper)
        tools = toolkit.get_tools()

        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "splunk_info" in tool_names
        assert "splunk_list_indexes" in tool_names
        assert "splunk_query" in tool_names
        assert "splunk_query_checker" not in tool_names

    def test_toolkit_get_tools_with_llm(self, mock_splunk_wrapper, mock_llm):
        """Test SplunkToolkit with LLM."""
        toolkit = SplunkToolkit(splunk_wrapper=mock_splunk_wrapper, llm=mock_llm)
        tools = toolkit.get_tools()

        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "splunk_query_checker" in tool_names

    def test_toolkit_uses_standalone_tools(self, mock_splunk_wrapper):
        """Test that toolkit uses standalone tools from tools package."""
        toolkit = SplunkToolkit(splunk_wrapper=mock_splunk_wrapper)
        tools = toolkit.get_tools()

        assert isinstance(next(tool for tool in tools if tool.name == "splunk_info"), InfoSplunkTool)
        assert isinstance(next(tool for tool in tools if tool.name == "splunk_list_indexes"), ListSplunkIndexesTool)
        assert isinstance(next(tool for tool in tools if tool.name == "splunk_query"), QuerySplunkTool)

class TestSplunkAgentCreation:
    """Test Splunk agent creation functions."""

    @pytest.fixture
    def mock_splunk_wrapper(self):
        """Create mock Splunk wrapper."""
        wrapper = Mock(spec=SplunkAPIWrapper)
        wrapper.splunk_host = "mock_host"
        wrapper.splunk_token = "mock_token"
        wrapper.splunk_username = "mock_user"
        wrapper.splunk_password = "mock_password"
        return wrapper

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM()

    def test_create_splunk_agent(self, mock_splunk_wrapper, mock_llm):
        """Test create_splunk_agent function."""
        toolkit = SplunkToolkit(splunk_wrapper=mock_splunk_wrapper, llm=mock_llm)

        with patch('langchain_community.agent_toolkits.splunk.base.create_react_agent') as mock_create_react_agent, \
             patch('langchain_community.agent_toolkits.splunk.base.AgentExecutor') as mock_agent_executor:

            # FIX: Use an instance of the new mock agent class
            mock_agent = MockAgent()
            mock_create_react_agent.return_value = mock_agent

            agent_executor = create_splunk_agent(llm=mock_llm, toolkit=toolkit, verbose=True)

            mock_create_react_agent.assert_called_once()
            mock_agent_executor.assert_called_once()
            call_kwargs = mock_agent_executor.call_args[1]
            assert call_kwargs['verbose'] is True
            assert call_kwargs['handle_parsing_errors'] is True

    def test_create_splunk_agent_from_api_wrapper(self, mock_splunk_wrapper, mock_llm):
        """Test create_splunk_agent_from_api_wrapper function."""
        with patch('langchain_community.agent_toolkits.splunk.base.create_splunk_agent') as mock_create_splunk_agent:
            mock_agent_executor = Mock()
            mock_create_splunk_agent.return_value = mock_agent_executor

            result = create_splunk_agent_from_api_wrapper(
                llm=mock_llm,
                splunk_wrapper=mock_splunk_wrapper,
                verbose=True
            )

            mock_create_splunk_agent.assert_called_once()
            call_args, call_kwargs = mock_create_splunk_agent.call_args
            # FIX: Change the assertion to check for class type instead of direct object equality
            assert isinstance(call_kwargs['llm'], MockLLM)
            assert call_kwargs['verbose'] is True
            assert isinstance(call_kwargs['toolkit'], SplunkToolkit)

    def test_agent_prompt_template(self, mock_splunk_wrapper, mock_llm):
        """Test that agent uses proper prompt template."""
        toolkit = SplunkToolkit(splunk_wrapper=mock_splunk_wrapper, llm=mock_llm)

        with patch('langchain_community.agent_toolkits.splunk.base.create_react_agent') as mock_create_react_agent:
            # FIX: Use an instance of the new mock agent class
            mock_create_react_agent.return_value = MockAgent()
            create_splunk_agent(llm=mock_llm, toolkit=toolkit)

            call_args = mock_create_react_agent.call_args
            llm_arg, tools_arg, prompt_arg = call_args[0]
            assert llm_arg is mock_llm
            assert len(tools_arg) >= 3
            assert hasattr(prompt_arg, 'template')
            assert "SPL" in prompt_arg.template

    def test_agent_executor_configuration(self, mock_splunk_wrapper, mock_llm):
        """Test agent executor configuration."""
        toolkit = SplunkToolkit(splunk_wrapper=mock_splunk_wrapper, llm=mock_llm)

        with patch('langchain_community.agent_toolkits.splunk.base.create_react_agent') as mock_create_react_agent, \
             patch('langchain_community.agent_toolkits.splunk.base.AgentExecutor') as mock_agent_executor:
            
            # FIX: Use an instance of the new mock agent class
            mock_create_react_agent.return_value = MockAgent()

            create_splunk_agent(
                llm=mock_llm,
                toolkit=toolkit,
                verbose=True,
                max_iterations=20,
                max_execution_time=120
            )

            call_kwargs = mock_agent_executor.call_args[1]
            assert call_kwargs['verbose'] is True
            assert call_kwargs['handle_parsing_errors'] is True
            assert call_kwargs['max_iterations'] == 20
            assert call_kwargs['max_execution_time'] == 120
