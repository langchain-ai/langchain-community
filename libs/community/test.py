from langchain_community.utilities.splunk import SplunkAPIWrapper
from langchain_openai import OpenAI
from langchain_community.agent_toolkits.splunk import create_splunk_agent_from_api_wrapper


splunk_wrapper = SplunkAPIWrapper(
    splunk_host="localhost",
    splunk_port=8089,
    splunk_token="eyJraWQiOiJzcGx1bmsuc2VjcmV0IiwiYWxnIjoiSFM1MTIiLCJ2ZXIiOiJ2MiIsInR0eXAiOiJzdGF0aWMifQ.eyJpc3MiOiJhZG1pbiBmcm9tIE1hY0Jvb2tQcm8uYXR0bG9jYWwubmV0Iiwic3ViIjoiYWRtaW4iLCJhdWQiOiJ0ZXN0IiwiaWRwIjoiU3BsdW5rIiwianRpIjoiMmE4NmVhZWI2OTUzZjM3ZTNiYmZhMTUxYzk5Mzg5ZDA2ZmU4MDdhNTYyYjkwYThhNDE5MzY4NGMyNWYyZTQxNCIsImlhdCI6MTc1NDUwMzU3NywiZXhwIjoxNzU0NzYyNzc3LCJuYnIiOjE3NTQ1MDM2Mzd9.n6WG-2nG0VuaV-up9nvb4TEC_1W17H9XZKyFH18fYzdZwNeMm7vz4zM7sOeqsAOwbSLK7W1phY_W950p7yeVzA",
    splunk_scheme="https",
    verify_ssl=False  # For self-signed certificates
)


# Initialize LLM
llm = OpenAI(temperature=0)

# Create agent
agent = create_splunk_agent_from_api_wrapper(
    llm=llm,
    splunk_wrapper=splunk_wrapper,
    verbose=True
)

# Use the agent
result=agent.invoke({"input": "what is number of error with status_code 200 in  web index in last 24 hours"})
print(result)
