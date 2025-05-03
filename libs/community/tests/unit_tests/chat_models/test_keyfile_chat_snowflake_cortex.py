# libs/community/tests/unit_tests/chat_models/test_keyfile_chat_snowflake_cortex.py

from langchain_community.chat_models import KeyfileChatSnowflakeCortex
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import KeyfileChatSnowflakeCortex

# ✅ Mocked dummy session and builder
class DummySession:
    def close(self):
        pass  # Prevent warning on __del__

class DummyBuilder:
    def configs(self, config):
        return self
    def create(self):
        return DummySession()

def test_chat_keyfile_snowflake_basic(monkeypatch):
    # ✅ Monkeypatch the Session.builder to use DummyBuilder
    monkeypatch.setattr(
        "langchain_community.chat_models.keyfile_chat_snowflake_cortex.Session.builder",
        DummyBuilder()
    )

    # ✅ Instantiate the model (should use dummy builder/session)
    chat = KeyfileChatSnowflakeCortex(
        snowflake_account="fake_account",
        snowflake_username="fake_user",
        private_key="fake_private_key_string",
        snowflake_warehouse="CORTEX_WH",
        snowflake_database="FAKE_DB",
        snowflake_schema="SCHEMA",
        snowflake_role="FAKE_ROLE",
        model="mistral-large",
        cortex_function="complete",
    )

    assert chat.model == "mistral-large"
