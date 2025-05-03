# test_keyfile_chat_snowflake_cortex.py

import pytest
from langchain_community.chat_models import KeyfileChatSnowflakeCortex
from langchain_core.messages import HumanMessage

import os

"""Test KeyfileChatSnowflakeCortex
Note: This test must be run with the following environment variables set:
    SNOWFLAKE_ACCOUNT="YOUR_SNOWFLAKE_ACCOUNT",
    SNOWFLAKE_USERNAME="YOUR_SNOWFLAKE_USERNAME",
    SNOWFLAKE_DATABASE="YOUR_SNOWFLAKE_DATABASE",
    SNOWFLAKE_SCHEMA="YOUR_SNOWFLAKE_SCHEMA",
    SNOWFLAKE_WAREHOUSE="YOUR_SNOWFLAKE_WAREHOUSE"
    SNOWFLAKE_ROLE="YOUR_SNOWFLAKE_ROLE",
    PRIVATE_KEY="YOUR_PRIVATE_KEY"    # base64-encoded DER string
"""

os.environ["SNOWFLAKE_ACCOUNT"] = "your_snowflake_account"
os.environ["SNOWFLAKE_USERNAME"] = "your_snowflake_username"
os.environ["SNOWFLAKE_WAREHOUSE"] = "your_snowflake_warehouse"
os.environ["SNOWFLAKE_DATABASE"] = "your_snowflake_database"
os.environ["SNOWFLAKE_SCHEMA"] = "your_snowflake_schema"
os.environ["SNOWFLAKE_ROLE"] = "your_snowflake_role"
os.environ["PRIVATE_KEY"]= "your_private_key"  # base64-encoded DER string


def test_chat_integration_with_cortex():
    chat = KeyfileChatSnowflakeCortex(
    snowflake_account=os.environ["SNOWFLAKE_ACCOUNT"],
    snowflake_username=os.environ["SNOWFLAKE_USERNAME"],
    private_key= os.environ["PRIVATE_KEY"],
    snowflake_warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    snowflake_database=os.environ["SNOWFLAKE_DATABASE"],
    snowflake_schema=os.environ["SNOWFLAKE_SCHEMA"],
    snowflake_role= os.environ["SNOWFLAKE_ROLE"],
    model="mistral-large",
    cortex_function="complete",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.95,
    )


    response = chat.invoke([HumanMessage(content="What's 2 + 2?")])
    assert isinstance(response.content, str)
    assert "4" in response.content or "four" in response.content.lower()
