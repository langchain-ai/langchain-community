"""
Splunk Agent Toolkit Example

This example demonstrates how to:
1. Set up Splunk connection with token authentication
2. Create and use a Splunk agent for natural language queries
3. Use standalone tools for specific tasks
4. Handle different types of queries and responses

Prerequisites:
- Splunk server with REST API enabled
- Valid Splunk authentication token
- Required environment variables set

Environment Variables:
- SPLUNK_HOST: Your Splunk server hostname
- SPLUNK_TOKEN: Your Splunk authentication token
- SPLUNK_PORT: Splunk port (default: 8089)
- OPENAI_API_KEY: Your OpenAI API key for the LLM
"""

import os
import sys
from typing import Optional

from langchain.llms import OpenAI
from langchain_community.agent_toolkits.splunk import (
    create_splunk_agent_from_api_wrapper,
    SplunkToolkit,
)
from langchain_community.tools.splunk import (
    InfoSplunkTool,
    ListSplunkIndexesTool,
    QuerySplunkTool,
    QueryCheckerSplunkTool,
)
from langchain_community.utilities.splunk import SplunkAPIWrapper


def setup_splunk_connection() -> Optional[SplunkAPIWrapper]:
    """Set up Splunk connection using environment variables."""
    
    # Get connection details from environment
    splunk_host = os.getenv("SPLUNK_HOST")
    splunk_token = os.getenv("SPLUNK_TOKEN")
    splunk_port = int(os.getenv("SPLUNK_PORT", "8089"))
    splunk_scheme = os.getenv("SPLUNK_SCHEME", "https")
    
    if not splunk_host:
        print("Error: SPLUNK_HOST environment variable not set")
        return None
        
    if not splunk_token:
        # Try fallback authentication
        splunk_username = os.getenv("SPLUNK_USERNAME")
        splunk_password = os.getenv("SPLUNK_PASSWORD")
        
        if not (splunk_username and splunk_password):
            print("Error: Either SPLUNK_TOKEN or both SPLUNK_USERNAME and SPLUNK_PASSWORD must be set")
            return None
        
        print("Warning: Using username/password authentication. Token authentication is recommended.")
        
        return SplunkAPIWrapper(
            splunk_host=splunk_host,
            splunk_port=splunk_port,
            splunk_token="",  # Empty token to trigger basic auth
            splunk_username=splunk_username,
            splunk_password=splunk_password,
            splunk_scheme=splunk_scheme,
            verify_ssl=False  # Often needed for self-signed certificates
        )
    
    return SplunkAPIWrapper(
        splunk_host=splunk_host,
        splunk_port=splunk_port,
        splunk_token=splunk_token,
        splunk_scheme=splunk_scheme,
        verify_ssl=False  # Set to True in production with valid certificates
    )


def demonstrate_standalone_tools(splunk_wrapper: SplunkAPIWrapper):
    """Demonstrate using individual Splunk tools."""
    
    print("\n" + "="*60)
    print("STANDALONE TOOLS DEMONSTRATION")
    print("="*60)
    
    # Create individual tools
    info_tool = InfoSplunkTool(splunk_wrapper=splunk_wrapper)
    indexes_tool = ListSplunkIndexesTool(splunk_wrapper=splunk_wrapper)
    query_tool = QuerySplunkTool(splunk_wrapper=splunk_wrapper)
    
    try:
        # 1. Environment Information
        print("\nGetting Splunk Environment Information:")
        print("-" * 40)
        info_result = info_tool.run("")
        print(info_result)
        
        # 2. List Indexes
        print("\nListing Available Indexes:")
        print("-" * 40)
        indexes_result = indexes_tool.run("")
        print(indexes_result)
        
        # 3. Execute Query
        print("\nExecuting Sample SPL Query:")
        print("-" * 40)
        sample_query = "search * | head 3 | table _time, host, sourcetype"
        print(f"Query: {sample_query}")
        query_result = query_tool.run(sample_query)
        print(query_result)
        
    except Exception as e:
        print(f"Error in standalone tools demo: {e}")


def demonstrate_query_checker(splunk_wrapper: SplunkAPIWrapper):
    """Demonstrate query validation tool."""
    
    print("\n" + "="*60)
    print("QUERY VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Try to create LLM for query checker
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API key not set. Skipping LLM-based query validation.")
        return
    
    try:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        checker_tool = QueryCheckerSplunkTool(splunk_wrapper=splunk_wrapper, llm=llm)
        
        # Test queries
        test_queries = [
            "search index=main",
            "search * | stats count by sourcetype",
            "search index=main | join host [search index=security]",  # Expensive join
            "search *",  # Potentially slow wildcard
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{1} Validating Query: {query}")
            print("-" * 50)
            try:
                validation_result = checker_tool.run(query)
                print(validation_result)
            except Exception as e:
                print(f"Validation error: {e}")
                
    except Exception as e:
        print(f"Error in query checker demo: {e}")


def create_and_demo_agent(splunk_wrapper: SplunkAPIWrapper) -> Optional[object]:
    """Create and demonstrate Splunk agent."""
    
    print("\n" + "="*60)
    print("SPLUNK AGENT DEMONSTRATION")
    print("="*60)
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("The agent requires an LLM to function.")
        return None
    
    try:
        # Initialize LLM
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        print("LLM initialized")
        
        # Create Splunk agent
        agent_executor = create_splunk_agent_from_api_wrapper(
            llm=llm,
            splunk_wrapper=splunk_wrapper,
            verbose=True,  # Show agent reasoning
            max_iterations=10,
            max_execution_time=60
        )
        
        print("Splunk agent created successfully")
        return agent_executor
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None


def run_agent_examples(agent_executor):
    """Run example queries with the Splunk agent."""
    
    example_queries = [
        # Environment discovery
        "What indexes are available in this Splunk environment?",
        
        # Basic searches
        "Show me the most recent 3 events from any index",
        
        # Statistical analysis
        "What are the top 3 sourcetypes by event count?",
        
        # Time-based search
        "Find any events from the last hour",
        
        # Error analysis
        "Look for any error or failure events in the data",
    ]
    
    print(f"\nRunning {len(example_queries)} example queries...")
    print("=" * 60)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)
        
        try:
            result = agent_executor.run(query)
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Ask user if they want to continue
        if i < len(example_queries):
            try:
                user_input = input("\nPress Enter to continue or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
            except KeyboardInterrupt:
                print("\nStopping example queries...")
                break


def interactive_mode(agent_executor):
    """Run agent in interactive mode."""
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Ask questions about your Splunk data in natural language!")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'help' for example queries.")
    print("-" * 60)
    
    help_examples = [
        "What indexes are available?",
        "Show me recent events from the main index",
        "Find error events in the last hour", 
        "What are the top sourcetypes?",
        "Show me events from host server1",
        "Find authentication failures",
        "What's the event count by index?"
    ]
    
    while True:
        try:
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_query.lower() == 'help':
                print("\nExample queries you can try:")
                for i, example in enumerate(help_examples, 1):
                    print(f"  {i}. {example}")
                continue
            elif not user_query:
                continue
            
            print(f"\nProcessing: {user_query}")
            result = agent_executor.run(user_query)
            print(f"\nResult:\n{result}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main execution function."""
    
    print("Splunk Agent Toolkit Example")
    print("=" * 60)
    
    # Environment setup check
    print("\n🔧 Environment Setup:")
    required_vars = ["SPLUNK_HOST", "SPLUNK_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("\nRequired environment variables:")
        print("- SPLUNK_HOST: Your Splunk server hostname")
        print("- SPLUNK_TOKEN: Your Splunk authentication token")  
        print("- OPENAI_API_KEY: Your OpenAI API key")
        print("\nOptional environment variables:")
        print("- SPLUNK_PORT: Splunk port (default: 8089)")
        print("- SPLUNK_SCHEME: http or https (default: https)")
        
        if not os.getenv("SPLUNK_TOKEN"):
            print("\nAlternative authentication (not recommended):")
            print("- SPLUNK_USERNAME: Splunk username")
            print("- SPLUNK_PASSWORD: Splunk password")
        
        return
    
    # Create Splunk connection
    try:
        splunk_wrapper = setup_splunk_connection()
        if not splunk_wrapper:
            return
        
        print("Splunk connection configured")
    except Exception as e:
        print(f"Failed to configure Splunk connection: {e}")
        return
    
    # Test connection
    print("\nTesting connection...")
    if not splunk_wrapper.test_connection():
        print("Failed to connect to Splunk server")
        print("Please check your host, port, and credentials.")
        return
    
    print("Successfully connected to Splunk")
    
    # Demo modes
    print("\nWhat would you like to do?")
    print("1. Demonstrate standalone tools")
    print("2. Demonstrate query validation") 
    print("3. Create and demo agent with example queries")
    print("4. Interactive mode with agent")
    print("5. All of the above")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return
    
    # Execute based on choice
    agent_executor = None
    
    if choice in ['1', '5']:
        demonstrate_standalone_tools(splunk_wrapper)
    
    if choice in ['2', '5']:
        demonstrate_query_checker(splunk_wrapper)
    
    if choice in ['3', '4', '5']:
        agent_executor = create_and_demo_agent(splunk_wrapper)
        
        if agent_executor and choice in ['3', '5']:
            run_agent_examples(agent_executor)
    
    if choice == '4' and agent_executor:
        interactive_mode(agent_executor)
    elif choice == '4' and not agent_executor:
        print("Agent creation failed. Cannot run interactive mode.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)
