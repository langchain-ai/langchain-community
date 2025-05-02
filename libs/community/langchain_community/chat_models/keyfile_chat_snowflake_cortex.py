"""Snowflake Cortex chat model with support for keyfile-based authentication."""

import base64
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from pydantic import Field, model_validator

from snowflake.snowpark import Session

from langchain_core.utils import get_from_dict_or_env
from langchain_community.chat_models.snowflake import ChatSnowflakeCortex, ChatSnowflakeCortexError


class KeyfileChatSnowflakeCortex(ChatSnowflakeCortex):
    """Snowflake Cortex chat model with keyfile authentication support."""

    private_key: Optional[str] = Field(default=None, exclude=True)
    private_key_path: Optional[str] = Field(default=None, exclude=True)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Sanitize and forward messages to Snowflake Cortex."""
        def sanitize(text):
            return (
                text.replace("’", "'")
                    .replace("‘", "'")
                    .replace('"', "'")
                    .replace("`", "'")
                    .replace("\\", "\\\\")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .replace("'", "''")
            )

        for msg in messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                msg.content = sanitize(msg.content)

        return super()._generate(messages, stop, run_manager, **kwargs)

    @classmethod
    @model_validator(mode="before")
    def validate_environment(cls, values: dict):
        """Load and validate environment variables and private key."""

        values["snowflake_username"] = get_from_dict_or_env(values, "snowflake_username", "SNOWFLAKE_USERNAME")
        values["snowflake_account"] = get_from_dict_or_env(values, "snowflake_account", "SNOWFLAKE_ACCOUNT")
        values["snowflake_database"] = get_from_dict_or_env(values, "snowflake_database", "SNOWFLAKE_DATABASE")
        values["snowflake_schema"] = get_from_dict_or_env(values, "snowflake_schema", "SNOWFLAKE_SCHEMA")
        values["snowflake_warehouse"] = get_from_dict_or_env(values, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE")
        values["snowflake_role"] = get_from_dict_or_env(values, "snowflake_role", "SNOWFLAKE_ROLE")

        private_key = values.get("private_key")
        private_key_path = values.get("private_key_path")

        if not private_key and private_key_path:
            try:
                with open(private_key_path, "rb") as key_file:
                    pem_data = key_file.read()
                    p_key = serialization.load_pem_private_key(
                        pem_data,
                        password=None,
                        backend=default_backend()
                    )
                    private_key_der = p_key.private_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                    private_key = base64.b64encode(private_key_der).decode("utf-8")
            except Exception as e:
                raise ValueError(f"Could not read/convert private key file: {e}")

        connection_params = {
            "account": values["snowflake_account"],
            "user": values["snowflake_username"],
            "private_key": private_key,
            "database": values["snowflake_database"],
            "schema": values["snowflake_schema"],
            "warehouse": values["snowflake_warehouse"],
            "role": values["snowflake_role"],
            "client_session_keep_alive": True,
        }

        try:
            values["session"] = Session.builder.configs(connection_params).create()
        except Exception as e:
            raise ChatSnowflakeCortexError(f"Failed to create session: {e}")

        return values
