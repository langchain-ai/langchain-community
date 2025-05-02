from typing import List, Optional, Any
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine, text
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import base64


class SnowflakeCortexEmbeddings(Embeddings, BaseModel):
    # Fields
    private_key: Optional[str] = Field(default=None, exclude=True)
    private_key_path: Optional[str] = Field(default=None, exclude=True)

    model: str = "e5-base-v2"
    embedding_function: str = "EMBED_TEXT_768"

    snowflake_username: str
    snowflake_account: str
    snowflake_database: str
    snowflake_schema: str
    snowflake_warehouse: str
    snowflake_role: str

    # Internal non-serializable field
    _engine: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict):
        # Load Snowflake config from env if not passed
        values["snowflake_username"] = get_from_dict_or_env(values, "snowflake_username", "SNOWFLAKE_USERNAME")
        values["snowflake_account"] = get_from_dict_or_env(values, "snowflake_account", "SNOWFLAKE_ACCOUNT")
        values["snowflake_database"] = get_from_dict_or_env(values, "snowflake_database", "SNOWFLAKE_DATABASE")
        values["snowflake_schema"] = get_from_dict_or_env(values, "snowflake_schema", "SNOWFLAKE_SCHEMA")
        values["snowflake_warehouse"] = get_from_dict_or_env(values, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE")
        values["snowflake_role"] = get_from_dict_or_env(values, "snowflake_role", "SNOWFLAKE_ROLE")

        # Load private key
        private_key = values.get("private_key")
        private_key_path = values.get("private_key_path")

        if not private_key and private_key_path:
            with open(private_key_path, "rb") as key_file:
                pem_data = key_file.read()
                p_key = serialization.load_pem_private_key(
                    pem_data, password=None, backend=default_backend()
                )
                private_key_der = p_key.private_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                private_key = base64.b64encode(private_key_der).decode("utf-8")

        if not private_key:
            private_key = get_from_dict_or_env(values, "private_key_str", "SNOWFLAKE_PRIVATE_KEY")

        values["private_key"] = private_key
        return values

    def __init__(self, **data):
        super().__init__(**data)

        # Initialize SQLAlchemy engine after validation
        private_key_bytes = base64.b64decode(self.private_key)
        self._engine = create_engine(
            URL(
                account=self.snowflake_account,
                user=self.snowflake_username,
                database=self.snowflake_database,
                schema=self.snowflake_schema,
                warehouse=self.snowflake_warehouse,
                role=self.snowflake_role,
            ),
            connect_args={"private_key": private_key_bytes},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        sql = text(f"SELECT SNOWFLAKE.CORTEX.{self.embedding_function}(:model, :text) AS embedding")
        embeddings = []
        with self._engine.connect() as conn:
            for text_input in texts:
                result = conn.execute(sql, {"model": self.model, "text": text_input})
                embeddings.append(result.fetchone()[0])
        return embeddings
