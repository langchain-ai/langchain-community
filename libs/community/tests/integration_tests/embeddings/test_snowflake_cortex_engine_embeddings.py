# test_snowflake_cortex_engine_embeddings.py

"""Test SnowflakeCortexEmbeddings
Note: This test must be run with the following environment variables set:
    SNOWFLAKE_ACCOUNT="YOUR_SNOWFLAKE_ACCOUNT",
    SNOWFLAKE_USERNAME="YOUR_SNOWFLAKE_USERNAME",
    SNOWFLAKE_DATABASE="YOUR_SNOWFLAKE_DATABASE",
    SNOWFLAKE_SCHEMA="YOUR_SNOWFLAKE_SCHEMA",
    SNOWFLAKE_WAREHOUSE="YOUR_SNOWFLAKE_WAREHOUSE"
    SNOWFLAKE_ROLE="YOUR_SNOWFLAKE_ROLE",
    PRIVATE_KEY="YOUR_PRIVATE_KEY"    # base64-encoded DER string
"""

import pytest
from langchain_community.embeddings import SnowflakeCortexEmbeddings
import os

os.environ["SNOWFLAKE_ACCOUNT"] = "your_snowflake_account"
os.environ["SNOWFLAKE_USERNAME"] = "your_snowflake_username"
os.environ["SNOWFLAKE_WAREHOUSE"] = "your_snowflake_warehouse"
os.environ["SNOWFLAKE_DATABASE"] = "your_snowflake_database"
os.environ["SNOWFLAKE_SCHEMA"] = "your_snowflake_schema"
os.environ["SNOWFLAKE_ROLE"] = "your_snowflake_role"
os.environ["PRIVATE_KEY"]= "your_private_key"  # base64-encoded DER string



def test_embed_query_returns_vector():
    embedder = SnowflakeCortexEmbeddings(
    private_key= os.environ["PRIVATE_KEY"],
    model="e5-base-v2",
    embedding_function="EMBED_TEXT_768",  # or another Cortex-supported function
    snowflake_account=os.environ["SNOWFLAKE_ACCOUNT"],
    snowflake_username=os.environ["SNOWFLAKE_USERNAME"],
    snowflake_warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    snowflake_database=os.environ["SNOWFLAKE_DATABASE"],
    snowflake_schema=os.environ["SNOWFLAKE_SCHEMA"],
    snowflake_role= os.environ["SNOWFLAKE_ROLE"],
    )

    result = embedder.embed_query("LangChain is awesome.")
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
    assert len(result) == 768


