import pytest
from langchain_core.runnables import RunnableLambda

from langchain_community.callbacks.mermaid_trace import MermaidTraceCallbackHandler


def test_mermaid_trace_integration() -> None:
    """Integration test for MermaidTraceCallbackHandler."""
    try:
        import mermaid_trace  # noqa: F401
    except ImportError:
        pytest.skip("mermaid-trace package not installed")

    handler = MermaidTraceCallbackHandler(host_name="IntegrationTestHost")
    
    # Create a simple chain
    def simple_func(x: dict) -> dict:
        return {"result": x["input"] + " world"}
    
    chain = RunnableLambda(simple_func)
    
    # Run the chain with the callback
    result = chain.invoke({"input": "hello"}, config={"callbacks": [handler]})
    
    assert result == {"result": "hello world"}
    # If we reached here without exception, the integration is working
    # (at least it doesn't crash on real library calls)

def test_mermaid_trace_error_integration() -> None:
    """Test error handling with real mermaid-trace library."""
    try:
        import mermaid_trace  # noqa: F401
    except ImportError:
        pytest.skip("mermaid-trace package not installed")

    handler = MermaidTraceCallbackHandler(host_name="ErrorTestHost")
    
    def error_func(x: dict) -> dict:
        raise ValueError("Integration test error")
    
    chain = RunnableLambda(error_func)
    
    with pytest.raises(ValueError, match="Integration test error"):
        chain.invoke({"input": "hello"}, config={"callbacks": [handler]})
    
    # Verify stack is empty after error
    assert len(handler._participant_stack) == 0
